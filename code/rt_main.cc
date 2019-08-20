#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mach/mach_time.h>
#include <string.h>

#include <pthread.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>

#include <xmmintrin.h>
#include <emmintrin.h>

#include "OpenImageDenoise/oidn.h"

#include <mach/mach.h>

typedef __m128 m128;
typedef __m128i m128i;

#define Assert(Expression) if(!(Expression)) {abort();}
#define AssertNAN(Number) Assert(!(Number != Number));
#define AssertInf(Number) Assert((*(unsigned int *)((void *)&Number) != 0xff800000) && (*(unsigned int *)((void *)&Number) != 0x7f800000))

static uint64_t numtris;
static uint64_t numprimaryrays;
static uint64_t numsplitpacketrays;
static uint64_t numsecondaryrays;
static uint64_t numrays;
static uint64_t numraytritests;
static uint64_t numraytriintersections;
static uint64_t numboundingboxtests;
static uint64_t numboundingboxintersections;

static uint64_t g_numpackets;
static uint64_t g_numsplitpackets;

#define SIMD_WIDTH 4
#define CACHELINE_SIZE (1 << 6)

#import "rt_math.h"

#define STB_IMAGE_IMPLEMENTATION
#import "stb_image.h"

enum
{
    PrimTypeTri,
    PrimTypeSphere,
};

enum
{
    NodeTypeInner,
    NodeTypeLeaf,
};

enum
{
    DimX,
    DimY,
    DimZ,
};

#define WIDTH_1 0
#define WIDTH_4 0

struct pixelbuffer_t
{
    int w, h;
    float *data;
    bool shouldprint;
    char *postfix;
};

static pixelbuffer_t g_pixelbuffers[4] = {};
// TODO: (Kapsy) Shove these into an enum.
#define COLOR_INDEX 0
#define ALBEDO_INDEX 1
#define NORMAL_INDEX 2
#define OUTPUT_INDEX 3
#define BUFFER_COUNT 4

// TODO: (Kapsy) Maybe best to put in a settings struct...
// NOTE: (Kapsy) We specify x and y so we can stratify.
static int g_rppx = 4;
static int g_rppy = 4;
static int g_rppcount = g_rppx*g_rppy;
#define BONUS_MOD 1

static float g_rppinv = 1.f/(float)g_rppcount;

static float *g_stratangles;
static int g_stratanglecount;

// NOTE: (Kapsy) I would actually say this is a dumb idea, and should move back to the chunk thing.
// That way no multi thread cache contentions.
static float g_stratdimx = 1.f/(float)g_rppx;
static float g_stratdimy = 1.f/(float)g_rppy;

  //////////////////////////////////////////////////////////////////////////////
 //// p32 Relative Pointers ///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef int p32;

inline void *
_P32ToP (p32 &P32)
{
    if (P32 == 0)
    {
        return 0;
    }

    return ((void *) ((char *) &P32 + P32));
}

inline void
P32AssignNull (p32 &P32)
{
     P32 = 0;
}

inline void
_P32AssignP (p32 &P32, void *P)
{
    if(P)
    {
        long Offset = ((char *) P - (char *) &P32);
        Assert (Offset >= INT_MIN && Offset <= INT_MAX);
        P32 = (int) Offset;
    }
    else
    {
        P32AssignNull(P32);
    }
}

#define P32ToP(P32, Type) (Type *)_P32ToP(P32)
#define P32AssignP(P32, P) _P32AssignP(P32, (void *)P)

  //////////////////////////////////////////////////////////////////////////////
 //// Perlin Noise ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define PERLIN_N (1 << 8)
struct perlin
{
    int permx[PERLIN_N];
    int permy[PERLIN_N];
    int permz[PERLIN_N];

    v3 randvec[PERLIN_N];
    float scale;
};

static void
InitPerlin(perlin *per, float scale)
{
    per->scale = scale;

    int N = PERLIN_N;

#define RandBipolar() (-1 + 2.f*drand48())

    for (int i=0 ; i<N ; i++)
    {
        per->randvec[i] = Unit (V3 (RandBipolar(), RandBipolar(), RandBipolar()));

        per->permx[i] = i;
        per->permy[i] = i;
        per->permz[i] = i;
    }

#define PermuteAxis(axis, i) \
    { \
        int tar = int(drand48()*(i+1)); \
        int tmp = axis[i]; \
        axis[i] = axis[tar]; \
        axis[tar] = tmp; \
    }

    for (int i=(N-1) ; i>0 ; i--)
    {
        PermuteAxis (per->permx, i);
        PermuteAxis (per->permy, i);
        PermuteAxis (per->permz, i);
    }
}

static float
_GetNoise(perlin *per, v3 p)
{
    p = p*per->scale;
    float u = p.x - floor(p.x);
    float v = p.y - floor(p.y);
    float w = p.z - floor(p.z);

    int i = floor(p.x);
    int j = floor(p.y);
    int k = floor(p.z);

    v3 c[2][2][2];

    for (int di=0 ; di<2 ; di++)
    for (int dj=0 ; dj<2 ; dj++)
    for (int dk=0 ; dk<2 ; dk++)
        c[di][dj][dk] =
            per->randvec[per->permx[(i+di) & (PERLIN_N - 1)] ^
                         per->permy[(j+dj) & (PERLIN_N - 1)] ^
                         per->permz[(k+dk) & (PERLIN_N - 1)]];

    float uu = u*u*(3 - 2*u);
    float vv = v*v*(3 - 2*v);
    float ww = w*w*(3 - 2*w);

    float accum = 0;

    for (int i=0 ; i<2 ; i++)
    for (int j=0 ; j<2 ; j++)
    for (int k=0 ; k<2 ; k++)
    {
        v3 weightv = V3 (u - i, v - j, w - k);
        accum +=
            (i*uu + (1 - i)*(1 - uu))*
            (j*vv + (1 - j)*(1 - vv))*
            (k*ww + (1 - k)*(1 - ww))*Dot (c[i][j][k], weightv);

    }

    Assert(accum <= 1.f);

    return (accum);
}

static float
GetNoise(perlin *per, v3 p)
{
    float accum = 0.f;
    v3 tempp = p;
    float weight = 1.f;
    for (int i=0 ; i<12 ; i++)
    {
        accum += weight*_GetNoise (per, tempp);
        weight *= 0.5f;
        tempp = tempp*2.f;
    }

    return 0.5*(1 + sin(0.2*p.z + 6*fabs (accum) + 2.f));
}

  //////////////////////////////////////////////////////////////////////////////
 //// Perlin Noise ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct perlin4
{
    int permx[PERLIN_N];
    int permy[PERLIN_N];
    int permz[PERLIN_N];

    v3 randvec[PERLIN_N];
    float scale;
};

static void
InitPerlin4 (perlin4 *per, float scale)
{
    per->scale = scale;

    int N = PERLIN_N;

#define RndBi4() (-1 + 2.f*drand48())

    for (int i=0 ; i<N ; i++)
    {
        per->randvec[i] =
            Unit (V3 ( RndBi4(), RndBi4(), RndBi4()));

        per->permx[i] = i;
        per->permy[i] = i;
        per->permz[i] = i;
    }

#define PermuteAxis(axis, i) \
    { \
        int tar = int(drand48()*(i+1)); \
        int tmp = axis[i]; \
        axis[i] = axis[tar]; \
        axis[tar] = tmp; \
    }

    for (int i=(N-1) ; i>0 ; i--)
    {
        PermuteAxis (per->permx, i);
        PermuteAxis (per->permy, i);
        PermuteAxis (per->permz, i);
    }
}


static m128
_GetNoise4 (perlin4 *per, v34 p)
{
    p = p*_mm_set1_ps (per->scale);

    m128 _i = _mm_set_ps (
            floor(p.x[0]),
            floor(p.x[1]),
            floor(p.x[2]),
            floor(p.x[3]));
    m128 _j = _mm_set_ps (
            floor(p.y[0]),
            floor(p.y[1]),
            floor(p.y[2]),
            floor(p.y[3]));
    m128 _k = _mm_set_ps (
            floor(p.z[0]),
            floor(p.z[1]),
            floor(p.z[2]),
            floor(p.z[3]));

    m128 u = p.x - _i;
    m128 v = p.y - _j;
    m128 w = p.z - _k;

    unsigned int i[4];
    _mm_storeu_ps((float *)i, _i);

    unsigned int j[4];
    _mm_storeu_ps((float *)j, _j);

    unsigned int k[4];
    _mm_storeu_ps((float *)k, _k);

    v34 c[2][2][2];

    for (int di=0 ; di<2 ; di++)
    for (int dj=0 ; dj<2 ; dj++)
    for (int dk=0 ; dk<2 ; dk++) {{{

        m128i permask = _mm_set1_epi32 (PERLIN_N - 1);
        m128i _pxi = _mm_and_si128 (_mm_cvttps_epi32 (_i) + _mm_set1_epi32 (di), permask);
        m128i _pyi = _mm_and_si128 (_mm_cvttps_epi32 (_j) + _mm_set1_epi32 (dj), permask);
        m128i _pzi = _mm_and_si128 (_mm_cvttps_epi32 (_k) + _mm_set1_epi32 (dk), permask);

        unsigned int pxi[4];
        _mm_storeu_ps((float *)pxi, _pxi);

        unsigned int pyi[4];
        _mm_storeu_ps((float *)pyi, _pyi);

        unsigned int pzi[4];
        _mm_storeu_ps((float *)pzi, _pzi);

        v3 va =
            per->randvec[per->permx[pxi[0]] ^
                         per->permy[pyi[0]] ^
                         per->permz[pzi[0]]];

        v3 vb =
            per->randvec[per->permx[pxi[1]] ^
                         per->permy[pyi[1]] ^
                         per->permz[pzi[1]]];

        v3 vc =
            per->randvec[per->permx[pxi[2]] ^
                         per->permy[pyi[2]] ^
                         per->permz[pzi[2]]];

        v3 vd =
            per->randvec[per->permx[pxi[3]] ^
                         per->permy[pyi[3]] ^
                         per->permz[pzi[3]]];

        c[di][dj][dk] =
            V34 (
                    _mm_set_ps (va.x, vb.x, vc.x, vd.x),
                    _mm_set_ps (va.y, vb.y, vc.y, vd.y),
                    _mm_set_ps (va.z, vb.z, vc.z, vd.z)
                );

    }}}

    m128 three = _mm_set1_ps (3.f);
    m128 two = MM_TWO;

    m128 uu = u*u*(three - two*u);
    m128 vv = v*v*(three - two*v);
    m128 ww = w*w*(three - two*w);

    m128 accum = MM_ZERO;

    for (int _i=0 ; _i<2 ; _i++)
    for (int _j=0 ; _j<2 ; _j++)
    for (int _k=0 ; _k<2 ; _k++)
    {
        m128 i = _mm_set1_ps (_i);
        m128 j = _mm_set1_ps (_j);
        m128 k = _mm_set1_ps (_k);
        v34 weightv = V34 (u - i, v - j, w - k);
        accum +=
            (i*uu + (MM_ONE - i)*(MM_ONE - uu))*
            (j*vv + (MM_ONE - j)*(MM_ONE - vv))*
            (k*ww + (MM_ONE - k)*(MM_ONE - ww))*Dot (c[_i][_j][_k], weightv);

    }

    assert( AllBitsSet4 (accum <= MM_ONE));

    return (accum);
}


static m128
GetNoise4 (perlin4 *per, v34 p)
{
    m128 accum = MM_ZERO;
    v34 tempp = p;
    m128 weight = MM_ONE;
    for (int i=0 ; i<12 ; i++)
    {
        accum += weight*_GetNoise4 (per, tempp);
        weight *= MM_HALF;
        tempp = tempp*MM_TWO;
    }

    m128 x;
    x[0] = sin(0.2*p.z[0] + 6*fabs (accum[0]) + 2.f);
    x[1] = sin(0.2*p.z[1] + 6*fabs (accum[1]) + 2.f);
    x[2] = sin(0.2*p.z[2] + 6*fabs (accum[2]) + 2.f);
    x[3] = sin(0.2*p.z[3] + 6*fabs (accum[3]) + 2.f);

    return _mm_set1_ps (0.5)*(MM_ONE + x);
}

static float
GetNoise2(perlin *per, v3 p)
{
    float accum = 0.f;
    v3 tempp = p;
    float weight = 1.f;
    for (int i=0 ; i<7 ; i++)
    {
        accum += weight*_GetNoise (per, tempp);
        weight *= 0.5f;
        tempp = tempp*2.f;
    }

    return (Clamp01 (accum*2.f));
}

static float
GetNoise3(perlin *per, v3 p)
{
    float accum = 0.f;
    v3 tempp = p;
    float weight = 1.f;
    for (int i=0 ; i<2 ; i++)
    {
        accum += weight*_GetNoise (per, tempp);
        weight *= 0.5f;
        tempp = tempp*3.9f;
    }

    return (_GetNoise (per, p));
}

static float
GetNoise32(perlin *per, v3 p)
{
    float accum = 0.f;
    v3 tempp = p;
    float weight = 1.f;
    for (int i=0 ; i<7 ; i++)
    {
        accum += weight*_GetNoise (per, tempp);
        weight *= 0.5f;
        tempp = tempp*2.f;
    }

    return (Clamp01 (accum*2.f));
}

static perlin4 testperlin4;
static perlin testperlin;
static perlin testperlin2;

  //////////////////////////////////////////////////////////////////////////////
 //// Tyre Tracks /////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static float
GetTyreThing (v3 p)
{
    float width = 0.38;

    // Left hand front.
    float t1xmin = 1.34f + sin(p.z*0.3f)*0.2f;
    float t1xmax = t1xmin + width;
    float t1zmin = -1000.0f;
    float t1zmax = 3.2f;
    unsigned int t1xres = ((p.x >= t1xmin) && (p.x < t1xmax));
    unsigned int t1zres = ((p.z >= t1zmin) && (p.z < t1zmax));
    unsigned int t1 = (t1xres && t1zres);

    // Left hand rear.
    float t2xmin = 2.2f - width + sin(p.z*0.3f)*0.3f;
    float t2xmax = t2xmin + width;
    float t2zmin = -1000.0f;
    float t2zmax = -2.7f;
    unsigned int t2xres = ((p.x >= t2xmin) && (p.x < t2xmax));
    unsigned int t2zres = ((p.z >= t2zmin) && (p.z < t2zmax));
    unsigned int t2 = (t2xres && t2zres);

    // Right hand front.
    float t3xmin = -2.10f + sin(p.z*0.3f)*0.2f;
    float t3xmax = t3xmin + width;
    float t3zmin = -1000.0f;
    float t3zmax = 3.2f;
    unsigned int t3xres = ((p.x >= t3xmin) && (p.x < t3xmax));
    unsigned int t3zres = ((p.z >= t3zmin) && (p.z < t3zmax));
    unsigned int t3 = (t3xres && t3zres);

    // Right hand rear.
    float t4xmin = -1.45f - width + sin(p.z*0.3f)*0.3f;
    float t4xmax = t4xmin + width;
    float t4zmin = -1000.0f;
    float t4zmax = -2.7f;
    unsigned int t4xres = ((p.x >= t4xmin) && (p.x < t4xmax));
    unsigned int t4zres = ((p.z >= t4zmin) && (p.z < t4zmax));
    unsigned int t4 = (t4xres && t4zres);

    float res = (float)(!(t1 || t2 || t3 || t4));

    return (res);
}


  //////////////////////////////////////////////////////////////////////////////
 //// Object Types ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

enum texture_type
{
    TEX_CHECKER,
    TEX_PLAIN,
    TEX_BITMAP,
    TEX_NORMAL,
    TEX_PERLIN,
    TEX_PERLIN2,
    TEX_PERLIN3,
    TEX_PERLIN_NORMAL,
};

struct texbuf_t
{
    int w, h, cpp;
    unsigned char *e;
};

struct texture
{
    texture_type type;
    v3 albedo;

    // TODO: (Kapsy) Put these into a discriminated union.
    perlin *perlin;
    perlin4 *perlin4;

    // NOTE: (Kapsy) RGB texture buffer
    texbuf_t bufa;
};

#define MAX_TEXTURES (1 << 6)
static texture textures[MAX_TEXTURES];
static int texturecount = 0;

enum material_type
{
    MAT_LAMBERTIAN_REFLECTION_MAP,
    MAT_DUMB_BRDF,
    MAT_CAR_PAINT,
    MAT_LAMBERTIAN,
    MAT_METAL,
    MAT_METAL_DIR_COLOR,
    MAT_DIELECTRIC,
    MAT_BACKGROUND,
    MAT_SOLID,

    // NOTE: (Kapsy) Debug mats.
    MAT_NORMALS,
    MAT_WUV,
    MAT_SCATTER,
};

struct mat_t
{
    material_type type;
    texture *tex;
    texture *texnorm;
    float fuzz; // TODO: (Kapsy) Metal only - move to discriminated union.
    float refindex;
    float reflfactor;

    char *name;

    float Ns;     // specular exponent, 0-1000
    v3 Ka;        // ambient color, RGB
    v3 Kd;        // diffuse color, RGB
    v3 Ks;        // specular color, RGB
    v3 Ke;        // ??
    float Ni;
    float d;      // dissolved, transparency
    int illum;    // illumination model

    float ksbase; // where 0 < ksbase <= 1
    float ksmax;  // where ksbase < ksmax <= 1

    unsigned int depthbonus;
    unsigned int remdepth; // TODO: (Kapsy) Rename to totaldepth
};

union tri_t
{
    struct
    {
        int A, B, C;
    };

    int e[3];
};

struct object_t
{
    int tricount;
    int vertcount;
    int vertnormcount;
    int vertuvcount;
    int matcount;

    // NOTE: (Kapsy) All tricount length;
    tri_t *tris;
    tri_t *trivns;
    tri_t *trivts;
    int *trimats;
    v3 *norms;
    v3 *vertnorms;
    v3 *vertuvs;
    v3 *tangents;

    v3 *verts;
    mat_t *mats;

    rect3 aabb;

    int bgmatindex;
};

static object_t g_object;

struct light_t
{
    v3 col;
    v3 dir;
    float intensity;
    float pad;
};

struct pointlight_t
{
    v3 col;
    v3 p;
    float intensity;
    float pad;
};

static light_t g_light = { V3 (1.f), Unit (V3 (0.3, -0.8, 0.5)), 4.0f };

static float g_illum = 0.33f;
static v3 g_illumcol = V3 (1.f, 0.8f, 0.4f);

#define MAX_POINT_LIGHTS (1 << 4)
static pointlight_t g_pointlights[MAX_POINT_LIGHTS] = {};
static int g_pointlightscount = 0;

  //////////////////////////////////////////////////////////////////////////////
 //// Ray /////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// NOTE: (Kapsy) Bounding box objects - for debugging only.
// static object bbobjects[MAX_OBJECTS];
// int bbobjectcount;

// TODO: (Kapsy) Rename to hit_t
struct hitrec
{
    float dist;
    float u, v;
    int primref;
    int primtype;
};

struct hitrec4
{
    m128 dist;
    m128 u;
    m128 v;
    m128 hitmask;

    // TODO: (Kapsy) Make these into m128is!
    m128 primref;
    m128 primtype;
};

union ray
{
    // TODO: (Kapsy) Should remove this, never use and make it a struct.
    struct
    {
        v3 A, B;
        float thit; // used?
        int remdepth;
        int havebonus;
    };

    struct
    {
        v3 orig, dir;
        float dnu0;
        float dnu1;
        float dnu2;
    };
};

inline ray
Ray(const v3 &a, const v3 &b)
{
    ray res = { a, b };
    res.thit = MAXFLOAT;
    return (res);
}

inline v3
PointAt(ray *r, float t)
{
    v3 res = r->A + t*r->B;
    return (res);
}

union ray4
{
    struct
    {
        v34 A, B;
        // TODO: (Kapsy) Fix this!
        m128 thit;
        // m128 pad;

        int remdepth;
        int havebonus;

        int splitcount;
        int pad1;
    };

    struct
    {
        v34 orig, dir;
        // TODO: (Kapsy) Fix this!
        m128 dnu;
        //m128 pad;

        int dnu1;
        int dnu2;

        int pad00;
        int pad11;
    };

};

inline v34
PointAt4 (ray4 *r, m128 t)
{
    v34 res = r->A + t*r->B;
    return (res);
}

inline ray4
Ray4 (const v34 &a, const v34 &b)
{
    ray4 res = { a, b };
    return (res);
}


  //////////////////////////////////////////////////////////////////////////////
 //// Camera //////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct camera
{
    v3 origin, lowerleft, horiz, vert;
    v3 u, v, w;
    float lensrad;
};


inline ray
GetRay (camera *c, float s, float t, float stratangle)
{

#if 0
    // NOTE: (Kapsy) Random in unit disk.
    v3 rand;
    do
    {
        rand = 2.f*V3 (drand48(), drand48(), 0) - V3 (1,1,0);
    }
    while (Dot (rand, rand) >= 1.f);
    v3 rd = c->lensrad*rand;

#else
    // NOTE: (Kapsy) Depth/motion blur stratification.
    float mag = sqrt (drand48 ());
    v3 stratvector = V3 (cos (stratangle), sin (stratangle), 0.f)*mag;
    v3 rd = c->lensrad*stratvector;

#endif

    v3 offset = c->u*rd.x + c->v*rd.y;
    ray res = Ray (c->origin + offset, c->lowerleft + s*c->horiz + t*c->vert - c->origin - offset);
    return (res);
}

  //////////////////////////////////////////////////////////////////////////////
 //// Sphere //////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct sphere
{
    v3 center;
    float rad;
    int matindex;
};

#define MAX_SPHERES (1 << 6)
static sphere spheres[MAX_SPHERES];
static int spherecount;

static void
TraverseSpheres (ray *r, hitrec *hit)
{
    float tnear = EPSILON;
    float tfar = hit->dist; // TODO: (Kapsy) Should be thit, if we start using multiple boxes?

    for (int i=0 ; i<spherecount ; i++)
    {
        float rad = spheres[i].rad;
        v3 center = spheres[i].center;

        v3 oc = r->orig - center;

        float a = Dot (r->dir, r->dir);
        float b = Dot (oc, r->dir);
        float c = Dot (oc, oc) - rad*rad;
        float discriminant = b*b - a*c;

        int testdis = 0;
        if (discriminant > 0.f)
        {
            float t = (-b - sqrt(discriminant))/a;

            if (tnear < t && t < tfar)
            {
                tfar = t;

                hit->dist = t;
                hit->primref = i;
                hit->primtype = PrimTypeSphere;
            }

            t = (-b + sqrt(discriminant))/a;
            if (tnear < t && t < tfar)
            {
                tfar = t;

                hit->dist = t;
                hit->primref = i;
                hit->primtype = PrimTypeSphere;
            }
        }
    }
}

static void
TraverseSpheres4 (ray4 *r, hitrec4 *hit)
{
    m128 tnear = _mm_set1_ps (EPSILON);
    m128 tfar = hit->dist; // TODO: (Kapsy) Should be thit, if we start using multiple boxes?
    m128 all = _mm_set1_epi32(0xffffffff);

    for (int i=0 ; i<spherecount ; i++)
    {
        m128 rad = _mm_set1_ps (spheres[i].rad);
        v34 center = V34 (spheres[i].center);

        v34 oc = r->orig - center;

        m128 a = Dot (r->dir, r->dir);
        m128 b = Dot (oc, r->dir);
        m128 c = Dot (oc, oc) - rad*rad;
        m128 disc = b*b - a*c; // discriminant

        m128 dm = disc > _mm_set1_ps (0.f);
        m128 dminv = _mm_xor_ps (dm, all);

        if (HaveBitsSet (dm))
        {
            m128 discsqrt = _mm_sqrt_ps (disc);

            // NOTE: (Kapsy) Negative part.
            m128 t1 = (-b - discsqrt)/a;
            m128 m1 = _mm_and_ps ((tnear < t1), (t1 < tfar));
            m1 = _mm_and_ps (m1, dm);
            m128 m1through = _mm_xor_ps (m1, all);
            tfar = _mm_and_ps (tfar, m1through) + _mm_and_ps (t1, m1);
            hit->dist = _mm_and_ps (hit->dist, m1through) + _mm_and_ps (t1, m1);

            // NOTE: (Kapsy) Positive part.
            m128 t2 = (-b + discsqrt)/a;
            m128 m2 = _mm_and_ps ((tnear < t2), (t2 < tfar));
            m2 = _mm_and_ps (m2, dm);
            m128 m2through = _mm_xor_ps (m2, all);
            tfar = _mm_and_ps (tfar, m2through) + _mm_and_ps (t2, m2);
            hit->dist = _mm_and_ps (hit->dist, m2through) + _mm_and_ps (t2, m2);

            m128 mthrough = _mm_and_ps (m1through, m2through);
            m128 mthroughinv = _mm_xor_ps (mthrough, all);

            hit->primref =
                _mm_and_ps (hit->primref, mthrough) +
                _mm_and_ps (_mm_set1_ps (i), mthroughinv);

            hit->primtype =
                _mm_and_ps (hit->primtype, mthrough) +
                _mm_and_ps (_mm_set1_ps (PrimTypeSphere), mthroughinv);
        }
    }
}

  //////////////////////////////////////////////////////////////////////////////
 //// Simple Memory Pool //////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct mempool_t
{
    char *base;
    char *at;
    uint64_t remaining;
};

static mempool_t g_bspmempool;
static mempool_t g_screenmempool;

static void *
PoolAlloc (mempool_t *mempool, uint64_t size, unsigned int align)
{
    uint64_t offset = (*(uint64_t *)mempool->at & (align - 1));
    uint64_t used = size + offset;

    Assert (mempool->remaining >= used);

    void *result = (void *)(mempool->at + offset);

    mempool->at += used;
    mempool->remaining -= used;

    return (result);
}

// TODO: (Kapsy) Move to headers.
#import "rt_aa_bsp.cc"
#import "rt_obj_loader.cc"

// TODO: (Kapsy) Move to camera section.
inline float
GetAutofocusDistance (camera *c, object_t *object, fastbsp_t *fastbsp, int nx, int ny)
{
    // NOTE: (Kapsy) Send out 4 autofocus rays, choose the closest.
    // NOTE: (Kapsy) Assume no coherence, so just use single rays here.
    float nxf = (float)nx;
    float nyf = (float)ny;

    float afarearatio = 0.2f;

    float midx = nxf*0.5f;
    float midy = nyf*0.5f;

    float afhalfx = nxf*afarearatio*0.5f;
    float l = midx - afhalfx;
    float r = midx + afhalfx;

    float afhalfy = nyf*afarearatio*0.5f;
    float t = midy + afhalfy;
    float b = midy - afhalfy;

    // TODO: (Kapsy) v2i type.
    int afpoints[] = {l, t, r, t,
                      l, b, r, b};

    int afpointscount = 4;

    float minfocusdist = 20.f;

    for (int i=0 ; i<afpointscount ; i++)
    {
        int pointx = afpoints[i*2];
        int pointy = afpoints[i*2 + 1];

        float u = (float)pointx/(float)nx;
        float v = (float)pointy/(float)ny;

        // TODO: (Kapsy) A bit dumb, because we want to ignore DOF.
        // TODO: (Kapsy) Need a simple collision function here.
        ray r1 = GetRay (c, u, v, 0.f);

        hitrec hit = {};
        hit.dist = MAXFLOAT;

        TraverseTris (&r1, &object->aabb, &hit, fastbsp);
        // TODO: (Kapsy) Should check spheres here too!

        if (hit.dist < MAXFLOAT)
        {
            v3 p = (r1.orig + (hit.dist)*r1.dir);
            float pdist = Length (p - r1.orig);
            // So targetfocusdist isn't getting updated.
            if (pdist < minfocusdist)
            {
                minfocusdist = pdist;
            }
        }
    }

    return (minfocusdist);
}

inline v3
RandomInUnitSphere()
{
    v3 v;
    do
    {
        v = 2.f*V3 (drand48(), drand48(), drand48()) - V3 (1);
    }
    while (SquaredLen (v) >= 1.f);
    return (v);
}

inline v34
RandomInUnitSphere4 ()
{
    m128 one = _mm_set1_ps (1.f);
    m128 two = _mm_set1_ps (2.f);

    v34 v;
    do
    {
        // TODO: (Kapsy) Should be a faster way to deal with these.
        float x0 = (drand48());
        float x1 = (drand48());
        float x2 = (drand48());
        float x3 = (drand48());

        float y0 = (drand48());
        float y1 = (drand48());
        float y2 = (drand48());
        float y3 = (drand48());

        float z0 = (drand48());
        float z1 = (drand48());
        float z2 = (drand48());
        float z3 = (drand48());

        m128 x = _mm_set_ps (x0, x1, x2, x3);
        m128 y = _mm_set_ps (y0, y1, y2, y3);
        m128 z = _mm_set_ps (z0, z1, z2, z3);

        v = two*V34 (x, y, z) - V34 (one);
    }
    while (HaveBitsSet (SquaredLen4 (v) >= one));
    return (v);
}

union randomunion_t
{
    m128i epi32;
    m128 ps;
};

typedef m128i randomseed_t;

static randomseed_t g_seed = _mm_set_epi32 (1234, 23316, 235402, 9443);
inline m128
RandomUnilateral4 (randomseed_t *seed)
{
    randomunion_t u = {};

    *seed = (*seed)*_mm_set1_epi32 (1103515245) + _mm_set1_epi32 (12345);
    u.epi32 = _mm_or_ps (_mm_and_si128 ((*seed), _mm_set1_epi32 (0x7fffff)), _mm_set1_epi32 (0x3f800000));

    m128 res = u.ps - MM_ONE;

    return (res);
}

inline m128
RandomBilateral4 (randomseed_t *seed)
{
    randomunion_t u = {};

    *seed = (*seed)*_mm_set1_epi32 (1103515245) + _mm_set1_epi32 (12345);
    u.epi32 = _mm_or_ps (_mm_and_si128 ((*seed), _mm_set1_epi32 (0x7fffff)), _mm_set1_epi32 (0x3f800000));

    m128 res = u.ps;

    return (res);
}


inline v34
RandomInUnitSphere4Fast (randomseed_t *seed)
{
    v34 v;
    do
    {
        // TODO: (Kapsy) Should be a faster way to deal with these.
       m128 x = RandomUnilateral4 (seed);
       m128 y = RandomUnilateral4 (seed);
       m128 z = RandomUnilateral4 (seed);

       v = MM_TWO*V34 (x, y, z) - V34 (MM_ONE);
    }

    while (HaveBitsSet (SquaredLen4 (v) >= MM_ONE));

    return (v);
}

inline v3
RandomInUnitSphereFast (randomseed_t *seed)
{
    v3 v;
    do
    {
        // TODO: (Kapsy) Should be a faster way to deal with these.
       m128 x = RandomUnilateral4 (seed);
       v = 2.f*V3 (x[0], x[1], x[2]) - V3 (1.f);
    }
    while (SquaredLen (v) >= 1.f);

    return (v);
}

// NOTE: (Kapsy) No matter what we do here the while loop seems to be faster!
inline v34
_RandomInUnitSphere4Fast (randomseed_t *seed)
{
    randomseed_t s = *seed;

    m128i c0 = _mm_set1_epi32 (1103515245);
    m128i c1 = _mm_set1_epi32 (12345);

    m128i a0 = _mm_set1_epi32 (0x7fffff);
    m128i a1 = _mm_set1_epi32 (0x3f800000);

    randomunion_t x = {};
    randomunion_t y = {};
    randomunion_t z = {};

    s = s*c0 + c1;
    x.epi32 = _mm_or_ps (_mm_and_si128 (s, a0), a1);

    s = s*c0 + c1;
    y.epi32 = _mm_or_ps (_mm_and_si128 (s, a0), a1);

    s = s*c0 + c1;
    z.epi32 = _mm_or_ps (_mm_and_si128 (s, a0), a1);

    x.ps -= MM_ONE;
    y.ps -= MM_ONE;
    z.ps -= MM_ONE;

    v34 v = V34 (MM_ZERO);
    v.x = MM_TWO*x.ps - MM_ONE;
    v.y = MM_TWO*y.ps - MM_ONE;
    v.z = MM_TWO*z.ps - MM_ONE;

    *seed = s;

    return (v);
}

// NOTE: (Kapsy) Dielectric functions.
#define Reflect(v, N) (v - 2*Dot (v, N)*N)

inline float
Schlick (float cos, float refindex) {
    float r0 = (1.f - refindex)/(1.f + refindex);
    r0 = r0*r0;
    r0 = r0 + (1.f - r0)*pow((1.f - cos), 5.f);

    return (r0);
}

inline m128
Schlick4 (m128 cos, m128 refindex) {

    m128 one = _mm_set1_ps (1.f);
    m128 five = _mm_set1_ps (5.f);

    m128 r0 = (one - refindex)/(one + refindex);
    r0 = r0*r0;
    r0 = r0 + (one - r0)*powf4((one - cos), five);

    return (r0);
}


static v34 GetAttenuation4 (texture *tex, m128 u, m128 v, v34 p);

static v3
GetAttenuation (texture *tex, float u, float v, v3 p)
{
    v3 res = V3 (1);

    switch (tex->type)
    {
        case TEX_CHECKER:
            {
                float m = 1.8f;
                float o = 0.6f;

                float selector = sin(m*p.x + o)*sin(m*p.z + o);

                if (selector > 0.f)
                    res = V3 (1.f, 0.5f, 1.f);
                else
                    res = V3 (0.35f, 0.f, 0.84f);

            } break;

        case TEX_BITMAP:
            {
                v34 cheapres = GetAttenuation4 (tex, _mm_set1_ps (u), _mm_set1_ps (v), V34 (p));
                res.x = cheapres.x[0];
                res.y = cheapres.y[0];
                res.z = cheapres.z[0];

            } break;

        case TEX_PLAIN:
            {
                res = tex->albedo;

            } break;

        case TEX_PERLIN:
            {
                res = V3 (1.0)*GetNoise (tex->perlin, p)*GetTyreThing(p);
                res = res*tex->albedo;

            } break;

        case TEX_PERLIN2:
            {
                // TODO: (Kapsy) Slow way for now.
                res = V3 (1.0)*GetNoise2 (tex->perlin, p);
                res = res*tex->albedo;

            } break;

        case TEX_NORMAL:
            {
            } break;

        case TEX_PERLIN_NORMAL:
            {
                v34 cheapres = GetAttenuation4 (tex, _mm_set1_ps (u), _mm_set1_ps (v), V34 (p));
                res.x = cheapres.x[0];
                res.y = cheapres.y[0];
                res.z = cheapres.z[0];
            } break;


        case TEX_PERLIN3:
            {
            } break;

        default:
            {
            } break;
    }

    return (res);
}

static inline v34
LerpV34 (v34 a, m128 t, v34 b)
{
    m128 one = _mm_set1_ps (1.f);
    m128 ca = one - t;
    m128 cb = t;

    v34 res = V34 (a.x*ca + b.x*cb,
                   a.y*ca + b.y*cb,
                   a.z*ca + b.z*cb);

    return (res);
}


static v34
GetAttenuation4 (texture *tex, m128 u, m128 v, v34 p)
{
    v34 res;

    switch (tex->type)
    {
        case TEX_CHECKER:
            {
                float m = 1.8f;
                float o = 0.6f;

                float selector0 = sin(m*p.x[0] + o)*sin(m*p.z[0] + o);
                float selector1 = sin(m*p.x[1] + o)*sin(m*p.z[1] + o);
                float selector2 = sin(m*p.x[2] + o)*sin(m*p.z[2] + o);
                float selector3 = sin(m*p.x[3] + o)*sin(m*p.z[3] + o);

                res.x[0] = selector0 > 0.f ? 1.f : 0.35f;
                res.x[1] = selector1 > 0.f ? 1.f : 0.35f;
                res.x[2] = selector2 > 0.f ? 1.f : 0.35f;
                res.x[3] = selector3 > 0.f ? 1.f : 0.35f;

                res.y[0] = selector0 > 0.f ? 0.5f : 0.f;
                res.y[1] = selector1 > 0.f ? 0.5f : 0.f;
                res.y[2] = selector2 > 0.f ? 0.5f : 0.f;
                res.y[3] = selector3 > 0.f ? 0.5f : 0.f;

                res.z[0] = selector0 > 0.f ? 1.f : 0.84f;
                res.z[1] = selector1 > 0.f ? 1.f : 0.84f;
                res.z[2] = selector2 > 0.f ? 1.f : 0.84f;
                res.z[3] = selector3 > 0.f ? 1.f : 0.84f;

            } break;

       case TEX_PLAIN:
           {
               res.x = _mm_set1_ps (tex->albedo.x);
               res.y = _mm_set1_ps (tex->albedo.y);
               res.z = _mm_set1_ps (tex->albedo.z);

           } break;

       case TEX_BITMAP:
           {
               texbuf_t *buf = &tex->bufa;

#if 0
               // NOTE: (Kapsy) 2x2 bilinear test.
               texbuf_t testbuf = {};
               testbuf.w = 2;
               testbuf.h = 2;
               testbuf.cpp = 3;

               unsigned char cols[] = {
                   0xff, 0x0, 0x0, 0x0, 0xff, 0x0,
                   0x0, 0xff, 0x0, 0xff, 0x0, 0x0
               };

               testbuf.e = &cols[0];
               buf = &testbuf;
#endif

               if (buf->e) {

                   m128 zero = _mm_set1_ps (0.f);
                   m128 one = _mm_set1_ps (1.f);
                   m128 half = _mm_set1_ps (0.5f);
                   m128 epsilon = _mm_set1_ps (0.0001f);

                   m128 x = Clamp014(u)*_mm_set1_ps(buf->w);
                   m128 y = Clamp014(one - v)*_mm_set1_ps(buf->h) - epsilon;

                   int w = buf->w;
                   int h = buf->h;
                   int cpp = buf->cpp;

                   v34 acol = V34 (zero);
                   v34 bcol = V34 (zero);
                   v34 ccol = V34 (zero);
                   v34 dcol = V34 (zero);

                   // TODO: (Kapsy) Make a uv4 struct.
                   v34 texelposi = V34 ( _mm_cvtepi32_ps( _mm_cvttps_epi32(x)),
                                         _mm_cvtepi32_ps( _mm_cvttps_epi32(y)), zero);
                   v34 texelpos = V34 (x, y, zero) - texelposi;

                   v34 lerpoffset = V34 (zero);

                   for (int i=0 ; i<SIMD_WIDTH ; i++)
                   {
                       // NOTE: (Kapsy) Pull in pixels based on position relative to center.
                       int xstride = 0;
                       int ystride = 0;

                       lerpoffset.x[i] = 0.5;
                       lerpoffset.y[i] = 0.5;

                       if (texelpos.x[i] < 0.5f)
                       {
                           xstride = -1;
                           lerpoffset.x[i] = -0.5;
                       }

                       if (texelpos.y[i] < 0.5f)
                       {
                           ystride = -1;
                           lerpoffset.y[i] = -0.5;
                       }

                       // TODO: (Kapsy) These should really be v2is.
                       // Could make a v34 to SIMD the clamp.

                       v3 texela = V3 (texelposi.x[i] + xstride, texelposi.y[i] + ystride, 0.0);
                       v3 texelb = V3 (texela.x + 1, texela.y, 0.0);
                       v3 texelc = V3 (texela.x, texela.y + 1, 0.0);
                       v3 texeld = V3 (texela.x + 1, texela.y + 1, 0.0);

#define ClampTexel(texela, w, h) \
                       { \
                           texela.x = Clamp (0, (w - 1), texela.x); \
                           texela.y = Clamp (0, (h - 1), texela.y); \
                       } \


                       ClampTexel (texela, w, h);
                       ClampTexel (texelb, w, h);
                       ClampTexel (texelc, w, h);
                       ClampTexel (texeld, w, h);

                       unsigned char *a = buf->e + (int)texela.y*w*cpp + (int)texela.x*cpp;
                       unsigned char *b = buf->e + (int)texelb.y*w*cpp + (int)texelb.x*cpp;
                       unsigned char *c = buf->e + (int)texelc.y*w*cpp + (int)texelc.x*cpp;
                       unsigned char *d = buf->e + (int)texeld.y*w*cpp + (int)texeld.x*cpp;

#define Col32ToV34(a, i, acol) \
                       { \
                           acol.r[i] = (float)a[0]; \
                           acol.g[i] = (float)a[1]; \
                           acol.b[i] = (float)a[2]; \
                       } \

                       Col32ToV34 (a, i, acol);
                       Col32ToV34 (b, i, bcol);
                       Col32ToV34 (c, i, ccol);
                       Col32ToV34 (d, i, dcol);
                   }

                   m128 lerpx = texelpos.x - lerpoffset.x;
                   m128 lerpy = texelpos.y - lerpoffset.y;

                   v34 abcol = LerpV34 (acol, lerpx, bcol);
                   v34 cdcol = LerpV34 (ccol, lerpx, dcol);
                   res = LerpV34 (abcol, lerpy, cdcol);

                   m128 u8maxinv = one/_mm_set1_ps((float)0xff);
                   res = res*u8maxinv;
               }
               else
               {
                   res.x = _mm_set1_ps (1.0f);
                   res.y = _mm_set1_ps (0.0f);
                   res.z = _mm_set1_ps (0.0f);
               }

           } break;

        case TEX_PERLIN:
            {
                for (int i=0 ; i<SIMD_WIDTH ; i++)
                {
                    v3 p0 = V3 (p.x[i], p.y[i], p.z[i]);

                    v3 res0 = V3 (1.0)*GetNoise (tex->perlin, p0)*GetTyreThing(p0);

                    res.x[i] = res0.x;
                    res.y[i] = res0.y;
                    res.z[i] = res0.z;
                }

                res = res*V34(tex->albedo);

            } break;

        case TEX_PERLIN2:
            {
                // slow way for now
                v3 p0 = V3 (p.x[0], p.y[0], p.z[0]);
                v3 res0 = V3 (1.0)*GetNoise2 (tex->perlin, p0);

                v3 p1 = V3 (p.x[1], p.y[1], p.z[1]);
                v3 res1 = V3 (1.0)*GetNoise2 (tex->perlin, p1);

                v3 p2 = V3 (p.x[2], p.y[2], p.z[2]);
                v3 res2 = V3 (1.0)*GetNoise2 (tex->perlin, p2);

                v3 p3 = V3 (p.x[3], p.y[3], p.z[3]);
                v3 res3 = V3 (1.0)*GetNoise2 (tex->perlin, p3);

                res.x[0] = res0.x;
                res.x[1] = res1.x;
                res.x[2] = res2.x;
                res.x[3] = res3.x;

                res.y[0] = res0.y;
                res.y[1] = res1.y;
                res.y[2] = res2.y;
                res.y[3] = res3.y;

                res.z[0] = res0.z;
                res.z[1] = res1.z;
                res.z[2] = res2.z;
                res.z[3] = res3.z;


                res = res*V34(tex->albedo);

            } break;

        case TEX_PERLIN3:
            {
                for (int i=0 ; i<SIMD_WIDTH ; i++)
                {
                    // NOTE: (Kapsy) Slow way for now.
                    v3 p0 = V3 (p.x[i], p.y[i], p.z[i]);
                    v3 res0 = V3 (1.0)*GetNoise3 (tex->perlin, p0);

                    res.x[i] = res0.x;
                    res.y[i] = res0.y;
                    res.z[i] = res0.z;
                }

                res = res*V34(tex->albedo);

            } break;

        case TEX_PERLIN_NORMAL:
            {
                // NOTE: (Kapsy) Slow way for now.

                float step = 0.1f;
                float a = 0.1f;

                for (int i=0 ; i<SIMD_WIDTH ; i++)
                {

                    // TODO: (Kapsy) Share this function so we just make a displacement map, which we then set to a normal texture with a post function. Will allow use to
                    v3 p0 = V3 (p.x[i], p.y[i], p.z[i]);
                    v3 basetexel = V3 (1.0)*GetNoise (tex->perlin, p0);

                    // TODO: (Kapsy) This could be problematic if our z is not constant.
                    // Might have to force it to a value.
                    v3 pi0 = V3 (p.x[i] + step, p.y[i], p.z[i]);
                    v3 pi1 = V3 (p.x[i] - step, p.y[i], p.z[i]);

                    v3 pj0 = V3 (p.x[i], p.y[i] + step, p.z[i]);
                    v3 pj1 = V3 (p.x[i], p.y[i] - step, p.z[i]);

                    float cn = 0.2f;
                    float npi0 = GetNoise (tex->perlin, pi0)*cn;
                    float npi1 = GetNoise (tex->perlin, pi1)*cn;
                    float npj0 = GetNoise (tex->perlin, pj0)*cn;
                    float npj1 = GetNoise (tex->perlin, pj1)*cn;

                    float tpi0 = GetTyreThing (pi0);
                    float tpi1 = GetTyreThing (pi1);
                    float tpj0 = GetTyreThing (pj0);
                    float tpj1 = GetTyreThing (pj1);

                    v3 S = V3 (1, 0, a*npi0*tpi0 - a*npi1*tpi1);
                    v3 T = V3 (0, 1, a*npj0*tpj0 - a*npj1*tpj1);

                    v3 SxT = Cross (S, T);
                    v3 N = SxT/Length (SxT);

                    // NOTE: (Kapsy) Normalize for texture
                    N.x = N.x*0.5 + 0.5f;
                    N.y = N.y*0.5 + 0.5f;
                    N.z = N.z*0.5f + 0.5f;

                    res.x[i] = N.x;
                    res.y[i] = N.y;
                    res.z[i] = N.z;
                }

            } break;

        case TEX_NORMAL:
            {
                Assert ("Do nothing here.");

            } break;
    }

    return (res);
}

#define MAX_DEPTH 2

enum
{
    MercMatBlackTrim,
    MercMatBlackTrim2,
    MercMatBody,
    MercMatChrome,
    MercMatChromeTrim,
    MercMatDarkChrome,
    MercMatGauges,
    MercMatGlass,
    MercMatHeadlampBulb,
    MercMatHeadlampLensBump,
    MercMatHeadlampLensFlat,
    MercMatLights,
    MercMatLogo,
    MercMatMirror,
    MercMatRed_Carpet,
    MercMatRedLeather,
    MercMatRedLeather2,
    MercMatRubber,
    MercMatRubberTrim,
    MercMatWhiteTrim,
    MercMatWinkerGlass,
    MercMatBathtub,
};

  //////////////////////////////////////////////////////////////////////////////
 //// Shading Functions ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define ApplyBonus(rnew, rold, mat) \
            rnew.remdepth = rold->remdepth - 1 + mat->depthbonus*rold->havebonus; \
            rnew.havebonus = 0; \


// TODO: (Kapsy) No reason we should have to pass matset mask here.
inline v34
GetTextureNormal (mat_t *mat, v34 p, v34 N0, v34 T, m128 u, m128 v, m128 matsetmask)
{
    v34 res = N0;

    if (mat->texnorm)
    {
        // NOTE: (Kapsy) Gram-Schmidt to get B.
        // Slightly concerned that we aren't dealing with handedness here.
        v34 B = Cross (N0, Unit (T - Dot (T, N0)*N0));

        m128 zero = MM_ZERO;
        m444 tangenttoworld = {
            T.x,  B.x,  N0.x,  zero,
            T.y,  B.y,  N0.y,  zero,
            T.z,  B.z,  N0.z,  zero,
            zero, zero, zero,  zero
        };

        v34 atten = GetAttenuation4 (mat->texnorm, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);
        v34 N1tangent = Unit (atten*V34 (MM_TWO) - V34 (MM_ONE));

        // NOTE: (Kapsy) N1 would be the derived N, N0 would be the geometry N
        // maybe, could get the flat N too, but don't really need for now.
        res = Unit (N1tangent*tangenttoworld);
    }

    return (res);
}

inline v3
GetTextureNormal (mat_t *mat, v3 p, v3 N0, v3 T, float u, float v)
{
    v3 res = N0;

    if (mat->texnorm)
    {
        // NOTE: (Kapsy) Gram-Schmidt to get B.
        // Slightly concerned that we aren't dealing with handedness here.
        v3 B = Cross (N0, Unit (T - Dot (T, N0)*N0));

        float zero = 0.f;
        m44 tangenttoworld = {
            T.x,  B.x,  N0.x,  zero,
            T.y,  B.y,  N0.y,  zero,
            T.z,  B.z,  N0.z,  zero,
            zero, zero, zero,  zero
        };

        v3 atten = GetAttenuation (mat->texnorm, u, v, p);
        v3 N1tangent = Unit (atten*2.f - V3 (1.f));

        // NOTE: (Kapsy) N1 would be the derived N, N0 would be the geometry N
        // maybe, could get the flat N too, but don't really need for now.
        res = Unit (N1tangent*tangenttoworld);
    }

    return (res);
}

// NOTE: (Kapsy) Obtain clamped vectors for both reflections and refractions.
//
// N0refl is the reflected geometry vector.
// N1refl is the reflected normal map vector.
// V is the reflected vector.
// Works like this:
// Obtain plane of Normal Map normal and reflected vector.
// Obtain plane of geometry.
// The cross product of these two planes always ensures a vector that lies on
// both planes, hence clamped.
//
// NOTE: (Kapsy) Another option I've read about is to flip the normal map
// normal, but that would look less natural, as there would be a jump from
// where the maximum valid reflection had been reached.
inline v34
ClampedReflection (v34 N0refl, v34 N1refl, v34 V)
{
    v34 reflplane = Unit (Cross (N1refl, V));
    v34 surfplane = Unit (N0refl);
    v34 clampedrefl = Unit (Cross (reflplane, surfplane));

    return (clampedrefl);
}

inline v3
ClampedReflection (v3 N0refl, v3 N1refl, v3 V)
{
    v3 reflplane = Unit (Cross (N1refl, V));
    v3 surfplane = Unit (N0refl);
    v3 clampedrefl = Unit (Cross (reflplane, surfplane));

    return (clampedrefl);
}

inline m128
ReflectionProb (mat_t *mat, v34 V, v34 N1refl, m128 VdotN1, m128 intmask, m128 extmask)
{
    // NOTE: (Kapsy) Obtain refraction color.
    m128 refrindex = _mm_set1_ps(mat->refindex);
    m128 lenVdotN1 = VdotN1/Length (V);

    m128 cos =
        _mm_and_ps(refrindex*lenVdotN1, intmask) +
        _mm_and_ps(MM_NEGONE*lenVdotN1, extmask);

    m128 niovernt =
        _mm_and_ps(refrindex, intmask) +
        _mm_and_ps(MM_ONE/refrindex, extmask);

    // Should already be UNIT???!!!
    v34 unitV = Unit (V);
    m128 dt = Dot (unitV, N1refl);
    m128 discriminant = MM_ONE - niovernt*niovernt*(MM_ONE - dt*dt);
    m128 discmask = (discriminant > MM_ZERO);

    // NOTE: (Kapsy) Obtain reflection probability.
    m128 reflfactor = _mm_set1_ps(mat->reflfactor);
    m128 reflprob =
        Clamp014( (_mm_and_ps (Schlick4 (cos, refrindex), discmask) +
                    _mm_and_ps (MM_ONE, MM_INV (discmask)))
                *reflfactor);

    return (reflprob);
}

inline float
ReflectionProb (mat_t *mat, v3 V, v3 N1refl, float VdotN1, bool external)
{
    // NOTE: (Kapsy) Obtain refraction color.
    float refrindex = mat->refindex;
    float lenVdotN1 = VdotN1/Length (V);

    float cos = refrindex*lenVdotN1;
    if (external)
        cos = -1.f*lenVdotN1;

    float niovernt = refrindex;
    if (external)
        niovernt = 1.f/refrindex;

    // Should already be UNIT???!!!
    v3 unitV = Unit (V);
    float dt = Dot (unitV, N1refl);
    float discriminant = 1.f - niovernt*niovernt*(1.f - dt*dt);

    // NOTE: (Kapsy) Obtain reflection probability.
    float reflprob = 1.f;
    if (discriminant > 0.f)
        reflprob = Schlick (cos, refrindex);

    reflprob = Clamp01 (reflprob*mat->reflfactor);

    return (reflprob);
}

struct colres_t
{
    v3 I; // Color
    v3 A; // Albedo
    v3 N; // Normal
};

struct colres4_t
{
    v34 I; // Color
    v34 A; // Albedo
    v34 N; // Normal
};

// NOTE: (Kapsy) Pretty sure this is a massive source of slowdown once we get to single rays, we still perform split checks.
static colres4_t GetColorForRaySplittingBySign (ray4 *r4, object_t *object, fastbsp_t *fastbsp, int depth, randomseed_t *seed);

// TODO: (Kapsy) Would like to generate this both for single and sse...
static v34
GetBackgroundColor (ray4 *r4, mat_t *mat)
{
    v34 I = V34 (MM_ZERO);

    // NOTE: (Kapsy) Create sky background coor.

    // NOTE: (Kapsy) No need to subtract origin, because the direction is already local.
    v34 dirlocal = Unit (r4->dir);

    // NOTE: (Kapsy) Longitude interpolate.
    //v34 hcola = V34 (V3 (0.96f, 0.65f, 0.34f)*1.0f);
    v34 hcola = V34 (V3 (0.96f, 0.45f, 0.34f)*0.9f);
    v34 hcolb = V34 (V3 (0.86f, 0.15f, 0.24f));

    m128 phi =
        _mm_set_ps(
                atan2(dirlocal.z[3], dirlocal.x[3]),
                atan2(dirlocal.z[2], dirlocal.x[2]),
                atan2(dirlocal.z[1], dirlocal.x[1]),
                atan2(dirlocal.z[0], dirlocal.x[0]));


    // NOTE: (Kapsy) Setting up sun.
    // Almost certain we have to do this before we remove the -ve value.
    // Maybe, add one?
    // So it becomes:
    // 2 - 1
    // 0 - 1
    // Need to confirm this though.
    // Sounds reasonable though.
    //
    // So in that space we can just add our angle, zero point
    // and then move back to 0-1 space?
    // if we add .5
    // it becomes
    // 2.5 - 1.5
    // 0.5 - 1.5
    // So this should be fine if we just modulo?
    // So we get:
    // 0.5 - 0.0 2.0 - 1.5
    // 0.5 - 1.5
    // So just need it to wrap around
    // Then the problem remains of flipping it.
    // For that we just go
    // 2a + b = 1, 1a + b = 0
    // 0a + b = 1, 1a + b = 0
    //
    // So
    //
    // b = 1
    // 1a + 1 = 0
    // a = -1
    //
    // or
    // a + b = 0
    // 2a + b = 1
    //
    // hmmmm
    // maybe
    // this is a polynomial
    // ax^2 + by + c
    // but easier to say
    //
    // x - 1:
    // 1  - 0
    // -1 -  0
    // sqrt(x*x)
    //
    // 1 - 0
    // 1 - 0

    m128 a0 = _mm_set1_ps (1.f/M_PI);
    phi = phi*a0;
    phi = phi + _mm_set1_ps (1.f);

    // NOTE: (Kapsy) Trying different sun angles.
    // phi = phi + _mm_set1_ps (1.58); // from front of car
    // phi = phi + _mm_set1_ps (0.6); // if we want to delay suns appearence a little
    phi = phi + _mm_set1_ps (0.4); // like this best so far. Need to see how animates.

    m128 wrapmask = (phi > _mm_set1_ps(2.f));
    phi = _mm_and_ps ((phi - _mm_set1_ps (2.f)), wrapmask) + _mm_and_ps (phi, MM_INV (wrapmask));
    phi = phi - _mm_set1_ps (1.f);

    m128 t0 = _mm_sqrt_ps(phi*phi);
    v34 hcol = (MM_ONE - t0)*hcola + t0*hcolb;

    // NOTE: (Kapsy) Latitude interpolate.
    m128 theta =
        _mm_set_ps(
                asin(dirlocal.y[3]),
                asin(dirlocal.y[2]),
                asin(dirlocal.y[1]),
                asin(dirlocal.y[0]));

    float rangemultiplier = 9.f;
    m128 range = _mm_set1_ps (M_PI*0.017453292519943*rangemultiplier);

    // NOTE: (Kapsy) Some munging of range to ensure that hcola
    // (sunny side) covers a greater vertical range than hcolb.
    range = range*_mm_set1_ps (0.55f) + range*_mm_set1_ps (0.45f)*(MM_ONE - t0);

    m128 a = (_mm_set1_ps(-1.f)/range);
    m128 b = MM_ONE;
    m128 t1 = Clamp014(theta*a + b);

    // NOTE: (Kapsy) Apply exp curve for more gradual transition.
    t1 = t1*t1;

    v34 vcola = V34 (V3 (0.45, 0.15, 0.8));

    v34 skycol = (MM_ONE - t1)*vcola + t1*hcol;

    // NOTE: (Kapsy) Plane test, get cloud noise.
    v34 po = V34 (MM_ZERO, _mm_set1_ps (10000.f), MM_ZERO);
    v34 n = V34 (Unit (V3(0.0, 1.0, 0.0)));

    m128 denom = Dot (n, r4->dir);

    // NOTE: (Kapsy) Create a "sun" bright spot...
    v34 suncol = V34 (V3 (0.96f, 0.65f, 0.34f)*0.8f);
    m128 sunt0 = (MM_ONE - t0);
    m128 sunt1 = t1;
    sunt0 = sunt0*sunt0*sunt0*sunt0*sunt0*sunt0;

    m128 sunt = sunt0*sunt1;
    sunt = sunt*sunt*sunt;

    // NOTE: (Kapsy) And an even brighter spot.
    v34 suncolbright = V34 (_mm_set1_ps (0.3f));
    m128 suntbright = sunt*sunt*sunt*sunt*sunt;

    skycol = skycol + suncol*sunt + suncolbright*suntbright;

    // USE SOMETHING OTHER THAN t1
    t1 = Dot ((po - r4->orig), n)/denom;

    m128 hitmaskp = _mm_and_ps((t1 >= MM_ZERO), (denom > _mm_set1_ps (1e-6)));
    v34 p = r4->orig + (t1)*r4->dir;

    // NOTE: (Kapsy) Stop drawing after 200km to avoid cloud acne.
    m128 maxd = _mm_set1_ps (200000.f);
    m128 d = Length ((t1)*r4->dir);
    hitmaskp = _mm_and_ps(hitmaskp, (d < maxd));

    v34 cloudmix = GetAttenuation4 (mat->tex, MM_ZERO, MM_ZERO, (p & hitmaskp));

    v34 cloudcol =
        (hcol + suncol*sunt)*(_mm_set1_ps(0.8) + _mm_set1_ps(0.12)*t0)
        + V34 (_mm_set1_ps (0.34) + _mm_set1_ps (0.0)*(MM_ONE - t0));

    // NOTE: (Kapsy) Mix sky and clouds.
    v34 hitcol = skycol*(MM_ONE - cloudmix.x) + cloudcol*cloudmix.x;
    I = (hitcol & hitmaskp) + (skycol & MM_INV (hitmaskp));

    return (I);
}


static v3
Refract (v3 I, v3 N, float ior)
{
    v3 n = N;

    float cosi = Clamp (-1.f, 1.f, Dot (N, I));

    // NOTE: (Kapsy) Index of refraction before entering the medium (air).
    float etai = 1;
    // NOTE: (Kapsy) Refraction index of the object that the ray has hit.
    float etat = ior;

    if (cosi < 0)
    {
        // Outside the surface.
        cosi = -cosi;
    }
    else
    {
        // Inside the surface, need to reverse the normal.
        n = -n;
        Swap (etai, etat);
    }

    float eta = etai/etat;
    float k = 1 - eta*eta*(1 - cosi*cosi);

    if (k < 0)
        return V3 (0);
    else
        return eta*I + (eta*cosi - sqrtf (k))*n;
}

// NOTE: (Kapsy) Just passing the single object for now, but should really pass a scene that contains lights etc.
inline colres_t
GetColorForRay (ray *r, object_t *object, fastbsp_t *fastbsp, randomseed_t *seed)
{
    colres_t res = {};

    hitrec hit = {};
    hit.dist = MAXFLOAT;

    TraverseTris (r, &object->aabb, &hit, fastbsp);

    TraverseSpheres (r, &hit);

    int matindex = 0;
    v3 N;
    // Tangent vector.
    v3 T = V3 (0.f);
    v3 p;

    float u;
    float v;

    if (hit.dist < MAXFLOAT)
    {
        int primtype = hit.primtype;
        int primref = hit.primref;
        if (primtype == PrimTypeTri)
        {
            matindex = object->trimats[primref];

            v3 A = V3 (0.f);
            v3 B = V3 (0.f);
            v3 C = V3 (0.f);
            float hitu = hit.u;
            float hitv = hit.v;
            float hitw = 1.f - hitu - hitv;

            v3 TA = V3 (0.f);
            v3 TB = V3 (0.f);
            v3 TC = V3 (0.f);

            int triindex = hit.primref;
            tri_t *trivn = object->trivns + triindex;

            // Vertex normals.
            A = object->vertnorms[trivn->A];
            B = object->vertnorms[trivn->B];
            C = object->vertnorms[trivn->C];

            // Tangents.
            TA = object->tangents[trivn->A];
            TB = object->tangents[trivn->B];
            TC = object->tangents[trivn->C];

            // Calculate actual surface norm.
            tri_t *tri = object->tris + triindex;
            v3 Av = object->verts[tri->A];
            v3 Bv = object->verts[tri->B];
            v3 Cv = object->verts[tri->C];

            v3 ABv = Bv - Av;
            v3 ACv = Cv - Av;
            v3 Nv = Unit (Cross (ABv, ACv)); // check order

            // Textures.
            tri_t *trivt = object->trivts + triindex;

            v3 uvA = object->vertuvs[trivt->A];
            v3 uvB = object->vertuvs[trivt->B];
            v3 uvC = object->vertuvs[trivt->C];

            float weightw = 1.f - (hitu + hitv);
            float weightu = 1.f - (hitw + hitv);
            float weightv = 1.f - (hitw + hitu);

            v3 uvres = weightw*uvA + weightu*uvB + weightv*uvC;

            u = uvres.e[0];
            v = uvres.e[1];

            // TODO: (Kapsy) Quicker to calc at point of collision?
            N = Unit (hitw*A + hitu*B + hitv*C);
            p = r->orig + hit.dist*r->dir;
            T = Unit (hitw*TA + hitu*TB + hitv*TC);
        }
        else if (primtype == PrimTypeSphere)
        {
            matindex = spheres[primref].matindex;


            int sphereindex = primref;
            sphere *s = spheres + sphereindex;

            float rad = s->rad;
            v3 center = s->center;

            p = r->orig + hit.dist*r->dir;
            N = (p - center)/rad;

            // TODO: (Kapsy) Need to think about this more, just going to cover top down for now.
            // Probably want to tie the coordinate system to the texture.
#if 1
            // TODO: (Kapsy) Not sure what this should be!
            T = V3 (1.f, 0.f, 0.f);

            // So if we have a (theta,phi) in spherical coordinates we just need to scale theta and phi to fractions.
            // If theta is the angle down from the pole, and phi is the angle around the axis through the poles, the normalization to [0,1] would be:
            // u = phi / (2*Pi)
            // v = theta / Pi

            v3 plocal = N;

            // TODO: (Kapsy) Figure out atan2, if more expensive, just use dot product and acos.
            float phi = atan2(plocal.z, plocal.x);
            float theta = asin(plocal.y);

            float pi = M_PI;
            float piinv = 1.f/pi;
            float half = 0.5f;
            float one = 1.f;

            u = one - (phi + pi)*(piinv*half);
            v = (theta + pi*half)*piinv;
#endif

        }
    }
    else
    {
        // NOTE: (Kapsy) Not hit, so the material becomes the background.
        matindex = object->bgmatindex;
    }


    mat_t *mat = object->mats + matindex;

    // Move this to settings.
    float bias = 1e-3;

    switch (mat->type)
    {
        case MAT_LAMBERTIAN:
            {
                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = GetTextureNormal (mat, p, N0, T, u, v);
                    v3 A = GetAttenuation (mat->tex, u, v, p);

                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);
                    //float negVdotN1 = Dot (negV, N1);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    // m128 extmask = (negVdotN0 > MM_ZERO);
                    // m128 intmask = MM_INV (extmask);

                    // rename to out???
                    v3 N0refl = N0;//(N0 & extmask) + (N0*MM_NEGONE & intmask);
                    v3 N1refl = N1;//(N1 & extmask) + (N1*MM_NEGONE & intmask);

                    float negVdotN1refl = Dot (negV, N1refl);

                    // NOTE: (Kapsy) Obtain clamped vectors for both reflections and refractions.
                    // NOTE: (Kapsy) Another option I've read about is
                    // to flip the normal map normal, but that would
                    // look less natural, as there would be a jump from
                    // where the maximum valid reflection had been
                    // reached.
                    v3 reflplane = Unit (Cross (N1refl, V));
                    v3 surfplane = Unit (N0refl);
                    v3 clampedrefl = Unit (Cross (reflplane, surfplane));

                    // NOTE: (Kapsy) Obtain reflection color.
                    v3 refl = N1 + RandomInUnitSphereFast (seed);
                    v3 reflres = refl;
                    if (Dot (refl, N0refl) < 0.f)
                    {
                        reflres = clampedrefl;
                    }

                    ray rrefl = Ray (p + N0refl*bias, reflres);
                    ApplyBonus (rrefl, r, mat);

                    colres_t col1 = GetColorForRay (&rrefl, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;


        case MAT_DIELECTRIC:
            {
                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = GetTextureNormal (mat, p, N0, T, u, v);

                    v3 A = V3 (1.f);
                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);
                    float negVdotN1 = Dot (negV, N1);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    //float extmask = (negVdotN0 > MM_ZERO);
                    //float intmask = MM_INV (extmask);
                    bool extmask = (negVdotN0 > 0.f);

                    // NOTE: (Kapsy) Normal for reflected geometry rays.
                    v3 N0refl = extmask ? N0 : N0*-1.f;
                    // NOTE: (Kapsy) Normal for reflected normal map rays.
                    v3 N1refl = extmask ? N1 : N1*-1.f;

                    float negVdotN1refl = Dot (negV, N1refl);

                    v3 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                    // Can't do this because we need values for refraction.
                    ///float reflprob = ReflectionProb (mat, V, N1refl, VdotN1, intmask, extmask);

                    // NOTE: (Kapsy) Obtain refraction color.
                    float refrindex = mat->refindex;
                    float lenVdotN1 = VdotN1/Length (r->dir);

                    float cos = extmask ? -1.f*lenVdotN1 : refrindex*lenVdotN1;

                    float niovernt = extmask ? 1.f/refrindex : refrindex;

                    v3 unitV = Unit (V);
                    float dt = Dot (unitV, N1refl);
                    float discriminant = 1.f - niovernt*niovernt*(1.f - dt*dt);
                    float discmask = (discriminant > 0.f);

                    // NOTE: (Kapys) Obtain the reflection probability.
                    float reflprob = Clamp01 ((discmask ? Schlick (cos, refrindex) : 1.f)*mat->reflfactor);

                    // NOTE: (Kapsy) Obtain refraction vector.
                    v3 refr = Unit (niovernt*(unitV - N1refl*dt) - N1refl*sqrt (discriminant));
                    if (Dot (refr, N0refl) > 0.f)
                        refr = clampedrefl;

                    // NOTE: (Kapsy) Obtain reflection vector.
                    v3 refl = Unit (V + negVdotN1refl*2.f*N1);
                    if (Dot (refl, N0refl) < 0.f)
                        refl = clampedrefl;

                    // NOTE: (Kapsy) Create new ray and get color.
                    v3 rnewdir;
                    v3 newbias;
                    m128 random = RandomUnilateral4 (seed);
                    if (random[0] <= reflprob)
                    {
                        rnewdir = refl;
                        newbias = N0refl*bias;
                    }
                    else
                    {
                        rnewdir = refr;
                        newbias = N0refl*-1.f*bias;
                    }

                    ray rnew = Ray (p + newbias, rnewdir);
                    ApplyBonus (rnew, r, mat);

                    colres_t col1 = GetColorForRay (&rnew, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;


        case MAT_METAL:
            {
                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = GetTextureNormal (mat, p, N0, T, u, v);

                    v3 A = GetAttenuation (mat->tex, u, v, p);

                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    bool extmask = (negVdotN0 > 0.f);

                    // NOTE: (Kapsy) Normal for reflected geometry rays.
                    v3 N0refl = N0;
                    // NOTE: (Kapsy) Normal for reflected normal map rays.
                    v3 N1refl = N1;

                    float negVdotN1refl = Dot (negV, N1refl);

                    v3 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                    // NOTE: (Kapsy) Obtain reflection vector.
                    v3 refl = Unit (V + negVdotN1refl*2.f*N1 + mat->fuzz*RandomInUnitSphereFast (seed));
                    if (Dot (refl, N0refl) < 0.f)
                        refl = clampedrefl;

                    ray rrefl = Ray (p + N0refl*bias, refl);
                    ApplyBonus (rrefl, r, mat);

                    colres_t col1 = GetColorForRay (&rrefl, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;

        case MAT_METAL_DIR_COLOR:
            {
                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = GetTextureNormal (mat, p, N0, T, u, v);

                    //v3 A = GetAttenuation (mat->tex, u, v, p);
                    v3 A = V3 (0.6f) + ((r->dir + V3 (1.f))*V3 (0.5f))*V3 (0.4f);

                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    bool extmask = (negVdotN0 > 0.f);

                    // NOTE: (Kapsy) Normal for reflected geometry rays.
                    v3 N0refl = N0;
                    // NOTE: (Kapsy) Normal for reflected normal map rays.
                    v3 N1refl = N1;

                    float negVdotN1refl = Dot (negV, N1refl);

                    v3 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                    // NOTE: (Kapsy) Obtain reflection vector.
                    v3 refl = Unit (V + negVdotN1refl*2.f*N1 + mat->fuzz*RandomInUnitSphereFast (seed));
                    if (Dot (refl, N0refl) < 0.f)
                        refl = clampedrefl;

                    ray rrefl = Ray (p + N0refl*bias, refl);
                    ApplyBonus (rrefl, r, mat);

                    colres_t col1 = GetColorForRay (&rrefl, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;

        case MAT_SOLID:
            {
                colres_t col0 = {};
                if (r->remdepth > 0)
                {
                    v3 A = GetAttenuation (mat->tex, u, v, p);
                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N + V3 (1.f))*V3 (0.5f);
                    }

                    col0.I = A;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;

        case MAT_LAMBERTIAN_REFLECTION_MAP:
            {
                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = GetTextureNormal (mat, p, N0, T, u, v);

                    v3 A = GetAttenuation (mat->tex, u, v, p);
                    float base = 0.2f;
                    float ks = base + A.x*(1.f - base);

                    // NOTE: (Kapsy) Set the "dirt" albedo.
                    A = V3(0.1, 0.1, 0.1);

                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    // Don't need to do this for reflection only!
                    bool extmask = (negVdotN0 > 0.f);

                    // rename to out???
                    // NOTE: (Kapsy) Normal for reflected geometry rays.
                    v3 N0refl = N0;//(N0 & extmask) + (N0*MM_NEGONE & intmask);
                    // NOTE: (Kapsy) Normal for reflected normal map rays.
                    v3 N1refl = N1;//(N1 & extmask) + (N1*MM_NEGONE & intmask);

                    float negVdotN1refl = Dot (negV, N1refl);

                    v3 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                    // NOTE: (Kapsy) Obtain reflection vector.
                    v3 refl = Unit ((V + negVdotN1refl*2.f*N1) + mat->fuzz*RandomInUnitSphereFast (seed));
                    if (Dot (refl, N0refl) < 0.f)
                        refl = clampedrefl;

                    // NOTE: (Kapsy) Obtain Lambert scatter vector.
                    v3 lamb = N1 + RandomInUnitSphereFast (seed);
                    if (Dot (lamb, N0refl) < 0.f)
                        lamb = clampedrefl;

                    v3 rnewdir = lamb;

                    float reflprob = ReflectionProb (mat, V, N1refl, VdotN1, extmask)*ks;
                    m128 random = RandomUnilateral4 (seed);
                    if (random[0] <= reflprob)
                    {
                        A = V3 (1.f);
                        rnewdir = refl;
                    }

                    // NOTE: (Kapsy) Create new ray and get color.
                    ray rnew = Ray (p + N0refl*bias, rnewdir);
                    ApplyBonus (rnew, r, mat);

                    colres_t col1 = GetColorForRay (&rnew, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;

        case MAT_DUMB_BRDF:
            {
                // the ratio of reflection of the specular term of incoming light.
                float ks = mat->Ks.r;
                // the "shininess" of the specularity.
                float ns = mat->Ns;

                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = GetTextureNormal (mat, p, N0, T, u, v);

                    v3 A = GetAttenuation (mat->tex, u, v, p);
                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);
                    //float negVdotN1 = Dot (negV, N1);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    // Don't need to do this for reflection only!
                   // float extmask = (negVdotN0 > );
                   // float intmask = MM_INV (extmask);

                    bool extmask = (negVdotN0 > 0.f);

                    // rename to out???
                    // NOTE: (Kapsy) Normal for reflected geometry rays.
                    v3 N0refl = N0;
                    // NOTE: (Kapsy) Normal for reflected normal map rays.
                    v3 N1refl = N1;

                    float negVdotN1refl = Dot (negV, N1refl);

                    v3 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                    // NOTE: (Kapsy) Obtain reflection vector.
                    v3 refl = Unit (V + negVdotN1refl*2.f*N1 + ns*RandomInUnitSphereFast (seed));
                    if (Dot (refl, N0refl) < 0.f)
                        refl = clampedrefl;

                    // NOTE: (Kapsy) Obtain Lambert scatter vector.
                    v3 lamb = N1 + RandomInUnitSphereFast (seed);
                    if (Dot (lamb, N0refl) < 0.f)
                        lamb = clampedrefl;

                    v3 rnewdir = lamb;

                    float reflprob = ks;
                    m128 random = RandomUnilateral4 (seed);
                    if (random[0] <= reflprob)
                    {
                        A = V3 (1.f);
                        rnewdir = refl;
                    }

                    // NOTE: (Kapsy) Create new ray and get color.
                    ray rnew = Ray (p + N0refl*bias, rnewdir);
                    ApplyBonus (rnew, r, mat);

                    colres_t col1 = GetColorForRay (&rnew, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;

        case MAT_CAR_PAINT:
            {
                colres_t col0 = {};

                if (r->remdepth > 0)
                {
                    v3 N0 = N;
                    v3 N1 = N;
                    //v3 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);

                    v3 A = GetAttenuation (mat->tex, u, v, p);
                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = A;
                        col0.N = (N1 + V3 (1.f))*V3 (0.5);
                    }

                    if (r->remdepth == mat->remdepth)
                    {
                        col0.A = V3 (0.1f);
                    }

                    v3 V = r->dir;

                    float VdotN0 = Dot (V, N0);
                    float VdotN1 = Dot (V, N1);

                    v3 negV = V*-1.f;

                    float negVdotN0 = Dot (negV, N0);

                    // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                    // Don't need to do this for reflection only!
                    bool extmask = (negVdotN0 > 0.f);


                    // NOTE: (Kapsy) Normal for reflected geometry rays.
                    v3 N0refl = N0;
                    // NOTE: (Kapsy) Normal for reflected normal map rays.
                    v3 N1refl = N1;

                    float negVdotN1refl = Dot (negV, N1refl);

                    // NOTE: (Kapsy) Obtain reflection vector.
                    v3 refl = Unit (V + negVdotN1refl*2.f*N1) + mat->Ns*RandomInUnitSphereFast (seed);

                    // NOTE: (Kapsy) Obtain Lambert scatter vector.
                    v3 lamb = N1 + RandomInUnitSphereFast (seed);

                    v3 rnewdir = lamb;

                    float reflprob = ReflectionProb (mat, V, N1refl, VdotN1, extmask);
                    //reflprob = 0.3;
                    m128 random = RandomUnilateral4 (seed);
                    if (random[0] <= reflprob)
                    {
                        A = V3 (1.f);
                        rnewdir = refl;
                    }

                    ray rnew = Ray (p + N0refl*bias, rnewdir);
                    ApplyBonus (rnew, r, mat);

                    colres_t col1 = GetColorForRay (&rnew, object, fastbsp, seed);
                    col0.I = A*col1.I;
                }

                res.I += col0.I;
                res.A += col0.A;
                res.N += col0.N;

            } break;

        case MAT_BACKGROUND:
            {
                ray4 r4 = {};
                r4.orig.x[0] = r->orig.x;
                r4.orig.y[0] = r->orig.y;
                r4.orig.z[0] = r->orig.z;

                r4.dir.x[0] = r->dir.x;
                r4.dir.y[0] = r->dir.y;
                r4.dir.z[0] = r->dir.z;

                v34 backcol = GetBackgroundColor (&r4, mat);
                v3 I = V3 (backcol.x[0], backcol.y[0], backcol.z[0]);

                res.I += I;
                res.A += I;
                res.N += V3 (0.f);

            } break;

        default:
            {

            } break;
    }

    return (res);
}

#define ApplyGlobalIllum4(base, change, illum) ((base)*illum*V34 (g_illumcol) + (change)*(_mm_set1_ps(1.f) - illum))

struct matset_t
{
    //m128 mask;
    unsigned int mask[SIMD_WIDTH];

    int count;
    int matindex;
    int primtype;
};


inline v34
_ClampReflected (v34 N, v34 phi)
{
    m128 x = Dot (N, phi);
    m128 xmask = (x >= MM_ZERO);
    v34 phiclamp = Unit (Cross (N, Cross (N, phi))*MM_NEGONE);
    v34 res = (phi & xmask) + (phiclamp & _mm_xor_ps (xmask, MM_ALL));

    return (res);
}

inline colres4_t
GetColorForRay4 (ray4 *r4, object_t *object, fastbsp_t *fastbsp, int depth, m128 outmask, v3 signs, randomseed_t *seed)
{
    // TODO: (Kapsy) Better to keep res on stack and then copy???
    __sync_fetch_and_add (&g_numpackets, 1);

    if (HaveBitsSet (outmask))
    {
        __sync_fetch_and_add (&g_numsplitpackets, 1);
    }

    // NOTE: (Kapsy) Might not be all that efficient to use these, especially the further we get down the stack.
    m128 zero = _mm_set1_ps (0.f);
    m128 half = _mm_set1_ps (0.5f);
    m128 one = _mm_set1_ps (1.f);
    m128 two = _mm_set1_ps (2.f);
    m128 all = _mm_set1_epi32 (0xffffffff);

    colres4_t res = {};
    hitrec4 hit4 = {};

    m128 maxfloat = _mm_set1_ps (MAXFLOAT);
    hit4.dist = maxfloat;

    TraverseTris4 (r4, &object->aabb, &hit4, outmask, signs, fastbsp);

    TraverseSpheres4 (r4, &hit4);

    // do our plane test here,

    m128 outmaskinv = _mm_xor_ps (outmask, all);

    m128 hitmask = (hit4.dist < maxfloat);
    m128 hitmaskinv = _mm_xor_ps (hitmask, all);

    matset_t matsets[SIMD_WIDTH] = {};
    int matsetcount = 0;

    if (HaveBitsSet (hitmaskinv))
    {
        // add a material set
        matset_t *matset = matsets + matsetcount++;

        _mm_storeu_ps ((float *) matset->mask, hitmaskinv);

        matset->matindex = object->bgmatindex;

        for (int i=0 ; i<SIMD_WIDTH ; i++)
        {
            if (hitmaskinv[i] != 0.f)
            {
                matset->count++;
            }
        }
        // TODO: (Kapsy) Set shading params p, N, d, u, v, w
    }

    // NOTE: (Kapsy)
    // These would all be separate matdefs (s: sphere, t: tri).
    // ___________
    // | mb | m1 |
    // |    | s  |
    // |____|____|
    // | m2 | m1 |
    // | t  | t  |
    // |____|____|

    // To create mat sets:
    // - Get inf hits
    // - Get tri hits, separated by material
    // - Get sphere hits, separated by material
    // - Process each as separate material sets
    //
    // Should result in:
    // A packet with all hits to a tri (or two) of same mat should only have to process normals once.
    // Same with spheres.
    // It would be more efficient to process norms for tri hits regardless, but that would depend on the material anyway. So best to do per material, per prim.

    // Life is great

    // NOTE: (Kapsy) Create mat sets for the packet.
    // NEED to take into account outmask!
    // Should rename it to splitmask.
    // Try and make these work aligned.
    unsigned int hitmaskstore[4];
    _mm_storeu_ps((float *)hitmaskstore, hitmask);

    for (int i=0 ; i<SIMD_WIDTH ; i++)
    {
        if (hitmaskstore[i] == 0xffffffff)
        {
            // this won't do, need to load based on type.
            // could we merge mats???
            // not easily, mats rely on 0 index

            int primtype = hit4.primtype[i];
            int matindex = 0;

            // don't like this, has to be a better way,,,
            // any reason we can't do this in Traverse???
            // store in hit....
        if (primtype == PrimTypeTri)
        {
            matindex = object->trimats[(int)hit4.primref[i]];
        }
        else if (primtype == PrimTypeSphere)
        {
            matindex = spheres[(int)hit4.primref[i]].matindex;
        }

            //matindex = MercMatRedLeather;
            //matindex = MercMatLambertTest;

            int matfound = 0;

            for (int m=0 ; m<matsetcount ; m++)
            {
                matset_t *matset = matsets + m;
                if ((matset->matindex == matindex) &&
                    (matset->primtype == primtype))
                {
                    matset->mask[i] = 0xffffffff;
                    matset->count++;
                    matfound = 1;
                    break;
                }
            }

            if (!matfound)
            {
                matset_t *matset = matsets + matsetcount++;

                // NOTE: (Kapsy) Not sure if this is the best way to do this.
                matset->mask[i] = 0xffffffff;

                matset->matindex = matindex;
                matset->primtype = primtype;
                matset->count++;
            }
        }
    }


    for (int i=0 ; i<matsetcount ; i++)
    {
        matset_t *matset = matsets + i;

        m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

        v34 N;
        // Tangent vector.
        v34 T = V34 (zero);

        //v34 Ns; // Surface Normal
        v34 p;
        m128 u;
        m128 v;

        m128 lena = zero;

        // NOTE: (Kapsy) Obtain N and p based on the primitive type.
        if (matset->primtype == PrimTypeTri)
        {
            v34 A4 = V34 (zero);
            v34 B4 = V34 (zero);
            v34 C4 = V34 (zero);
            m128 hitu = hit4.u;
            m128 hitv = hit4.v;
            m128 hitw = _mm_set1_ps (1.f) - hitu - hitv;

            v34 TA4 = V34 (zero);
            v34 TB4 = V34 (zero);
            v34 TC4 = V34 (zero);

#if 1
            // NOTE: Obtain the vertex normals only for active tris.
            // Worth noting, might use cross product depending on the shader.
            // TODO: (Kapsy) Should make this go wide.
            for (int j=0 ; j<SIMD_WIDTH ; j++)
            {
                if (matset->mask[j] == 0xffffffff)
                {
                    int triindex = (int)hit4.primref[j];
                    tri_t *trivn = object->trivns + triindex;

                    // Vertex normals.
                    v3 A = object->vertnorms[trivn->A];
                    v3 B = object->vertnorms[trivn->B];
                    v3 C = object->vertnorms[trivn->C];

                    A4.x[j] = A.x;
                    A4.y[j] = A.y;
                    A4.z[j] = A.z;

                    B4.x[j] = B.x;
                    B4.y[j] = B.y;
                    B4.z[j] = B.z;

                    C4.x[j] = C.x;
                    C4.y[j] = C.y;
                    C4.z[j] = C.z;


                    // check for normal texture here?
                    // Tangents.
                    v3 TA = object->tangents[trivn->A];
                    v3 TB = object->tangents[trivn->B];
                    v3 TC = object->tangents[trivn->C];

                    TA4.x[j] = TA.x;
                    TA4.y[j] = TA.y;
                    TA4.z[j] = TA.z;

                    TB4.x[j] = TB.x;
                    TB4.y[j] = TB.y;
                    TB4.z[j] = TB.z;

                    TC4.x[j] = TC.x;
                    TC4.y[j] = TC.y;
                    TC4.z[j] = TC.z;


                    // Calculate actual surface norm.

                    tri_t *tri = object->tris + triindex;
                    v3 Av = object->verts[tri->A];
                    v3 Bv = object->verts[tri->B];
                    v3 Cv = object->verts[tri->C];

                    v3 ABv = Bv - Av;
                    v3 ACv = Cv - Av;
                    v3 Nv = Unit (Cross (ABv, ACv)); // check order

                    //Ns.x[j] = Nv.x;
                    //Ns.y[j] = Nv.y;
                    //Ns.z[j] = Nv.z;

                    // Okay, to make this work, we need to calculate a TBN for each vector.
                    // We then interpolate just like we do with the vector normals to get the TBN for that point.
                    // Only need to do this for Normal/Textured tris so shouldn't be that much?
                    // Not entirely sure that this is the right way to do it, but it seems to be best for now.

                    // This:
                    //
                    // When tangent vectors are calculated on larger meshes
                    // that share a considerable amount of vertices the tangent
                    // vectors are generally averaged to give nice and smooth
                    // results when normal mapping is applied to these
                    // surfaces. A problem with this approach is that the three
                    // TBN vectors could end up non-perpendicular to each other
                    // which means the resulting TBN matrix would no longer be
                    // orthogonal. Normal mapping will be only slightly off
                    // with a non-orthogonal TBN matrix, but it’s still
                    // something we can improve.
                    //
                    // It is not necessary to store an extra array containing
                    // the per-vertex bitangent since the cross product N × T′
                    // can be used to obtain mB′, where m = ±1 represents the
                    // handedness of the tangent space. The handedness value
                    // must be stored per- vertex since the bitangent B′
                    // obtained from N × T′ may point in the wrong direc- tion.
                    // The value of m is equal to the determinant of the matrix
                    // in Equation (7.39). One may find it convenient to store
                    // the per-vertex tangent vector T′ as a four- dimensional
                    // entity whose w coordinate holds the value of m. Then the
                    // bitangent B′ can be computed using the formula
                    //
                    // B′ = T′w(N × T′)
                    //
                    // where the cross product ignores the w coordinate. This
                    // works nicely for vertex programs by avoiding the need to
                    // specify an additional array containing the per- vertex m
                    // values.
                    //
                    //
                    //
                    // Code that demonstrates how per-vertex tangent vectors
                    // can be calculated for an arbitrary mesh is shown in
                    // Listing 7.1 This code calculates the tangent and
                    // bitangent directions for each triangle in a mesh and
                    // adds them to a cumulative total for each vertex used by
                    // the triangle. It then loops over all vertices, or-
                    // thonormalizes the tangent and bitangent for each one,
                    // and outputs a single four- dimensional tangent vector
                    // for each vertex whose fourth coordinate contains a
                    // handedness value.
                    //
                    // So all we need is this tangent vector (and m).
                    // The only thing we have to watch out for is that we
                    // already have the normal for each vertex, so we need to
                    // calculate our TBN based off that and not the average.
                    //
                    // So think I will try somthing like this, AFTER the object file load:
                    //
                    // Go through all triangles, create TBNs for the triangles
                    // only if the material contains a normal map.
                    // Actually, we only need to create TBs for each triangle.
                    // So, for each vertex, we need a list of associated TBs.
                    // With that we can then average and create tangent vector + m.
                    // This would exist in the same way as vertex normals.
                    //
                    // Not sure though, do we calculate the bitangent for each
                    // vertex? I would assume so if that's what they're doing
                    // in the vertex shader.
                    // Then the T, B are interpolated along with the normal?
                    // Need to play with this.
                    // Either way we don't need to store the bitangent for each
                    // vertex, can calculate when needed.
                    // The other thing we could do is convert the incoming
                    // light/ray vectors into tangent space, do exactly what
                    // the book does, but would have to convert back again for
                    // reflection rays etc, so better to work in object space
                    // from the start.

                    // Textures.
                    tri_t *trivt = object->trivts + triindex;

                    v3 uvA = object->vertuvs[trivt->A];
                    v3 uvB = object->vertuvs[trivt->B];
                    v3 uvC = object->vertuvs[trivt->C];

                    // NOTE: (Kapsy) Idea is that we get 3 vals that we lerp on based on our local wuv, so it's just a weighting.
                    float weightw = 1.f - (hitu[j] + hitv[j]);
                    float weightu = 1.f - (hitw[j] + hitv[j]);
                    float weightv = 1.f - (hitw[j] + hitu[j]);

                    v3 uvres = weightw*uvA + weightu*uvB + weightv*uvC;

                    u[j] = uvres.e[0];
                    v[j] = uvres.e[1];
                }
            }

            u = _mm_and_ps (u, matsetmask);
            v = _mm_and_ps (v, matsetmask);

            // TODO: (Kapsy) Quicker to calc at point of collision?
            N = (Unit (hitw*A4 + hitu*B4 + hitv*C4)) & matsetmask;
            p = (r4->orig + (hit4.dist)*r4->dir) & matsetmask;
            T = (Unit (hitw*TA4 + hitu*TB4 + hitv*TC4)) & matsetmask;

#else
            for (int j=0 ; j<SIMD_WIDTH ; j++)
            {
                if (matset->mask[j] == 0xffffffff)
                {
                    int triindex = (int)hit4.primref[j];
                    //tri_t *trivn = object->trivns + triindex;
                    tri_t *tri = object->tris + triindex;

                    v3 A = object->verts[tri->A];
                    v3 B = object->verts[tri->B];
                    v3 C = object->verts[tri->C];

                    A4.x[j] = A.x;
                    A4.y[j] = A.y;
                    A4.z[j] = A.z;

                    B4.x[j] = B.x;
                    B4.y[j] = B.y;
                    B4.z[j] = B.z;

                    C4.x[j] = C.x;
                    C4.y[j] = C.y;
                    C4.z[j] = C.z;
                }
            }

            v34 AB4 = B4 - A4;
            v34 AC4 = C4 - A4;
            v34 N = Unit (Cross (AB4, AC4));
#endif

        }
        // TODO: (Kapsy) Maybe rename matsets to hitrecsets.
        else if (matset->primtype == PrimTypeSphere)
        {
            m128 rad4 = zero;
            v34 center4 = V34 (zero);

            for (int j=0 ; j<SIMD_WIDTH ; j++)
            {
                if (matset->mask[j] == 0xffffffff)
                {
                    int sphereindex = (int)hit4.primref[j];
                    sphere *s = spheres + sphereindex;

                    float rad = s->rad;
                    v3 center = s->center;

                    center4.x[j] = center.x;
                    center4.y[j] = center.y;
                    center4.z[j] = center.z;

                    rad4[j] = rad;
                }
            }

            p = (r4->orig + (hit4.dist)*r4->dir) & matsetmask;
            N = ((p - center4)/rad4) & matsetmask;

            // TODO: (Kapsy) Need to think about this more, just going to cover top down for now.
            // Probably want to tie the coordinate system to the texture.
#if 1
            //v34 B =
            // TODO: (Kapsy) Not sure what this should be!
            T = V34 (one, zero, zero);

            // So if we have a (theta,phi) in spherical coordinates we just need to scale theta and phi to fractions.
            // If theta is the angle down from the pole, and phi is the angle around the axis through the poles, the normalization to [0,1] would be:
            // u = phi / (2*Pi)
            // v = theta / Pi

            v34 plocal = N;

            // TODO: (Kapsy) Figure out atan2, if more expensive, just use dot product and acos.
            m128 phi =
                _mm_set_ps(
                        atan2(plocal.z[3], plocal.x[3]),
                        atan2(plocal.z[2], plocal.x[2]),
                        atan2(plocal.z[1], plocal.x[1]),
                        atan2(plocal.z[0], plocal.x[0]));

            m128 theta =
                _mm_set_ps(
                        asin(plocal.y[3]),
                        asin(plocal.y[2]),
                        asin(plocal.y[1]),
                        asin(plocal.y[0]));

            // m128 theta = asin(plocal.y);

            // NOTE: (Kapsy) Okay, it works fine, just that cos by itself does not give a linear position across the circle.
            // We need to convert it into radians.
            // Damn obvious when you think about what cos actually means and what is happening.
            // So quite likely PS solution _is_ faster because it doesn't use dot products etc.
            // We really don't care that much, probably not going to be texturing spheres everywhere.

            m128 pi = _mm_set1_ps (M_PI);
            m128 piinv = _mm_set1_ps (1.f/M_PI);

            u = one - (phi + pi)*(piinv*half);
            v = (theta + pi*half)*piinv;

            u = _mm_and_ps (u, matsetmask);
            v = _mm_and_ps (v, matsetmask);
#endif

        }


        // NOTE: (Kapsy) Set values not hit to something reasonable so math doesn't blow up.

        mat_t *mat = object->mats + matset->matindex;

#if 1
        switch (mat->type)
        {

            case MAT_LAMBERTIAN:
                {
                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    colres4_t col0 = {};

                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);

                        // TODO: (Kapsy) Do this in a way so we don't repeat everywhere.
                        v34 A = GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);
                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N1 + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }

                        // TODO: (Kapsy) Move this to options.
                        m128 bias = _mm_set1_ps(1e-3);

                        {
                            v34 V = r4->dir;

                            m128 VdotN0 = Dot (V, N0);
                            m128 VdotN1 = Dot (V, N1);

                            v34 negV = V*MM_NEGONE;

                            m128 negVdotN0 = Dot (negV, N0);
                            m128 negVdotN1 = Dot (negV, N1);

                            // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                            m128 extmask = (negVdotN0 > MM_ZERO);
                            m128 intmask = MM_INV (extmask);

                            // rename to out???
                            v34 N0refl = N0;//(N0 & extmask) + (N0*MM_NEGONE & intmask);
                            v34 N1refl = N1;//(N1 & extmask) + (N1*MM_NEGONE & intmask);

                            m128 negVdotN1refl = Dot (negV, N1refl);

                            // NOTE: (Kapsy) Obtain clamped vectors for both reflections and refractions.
                            // NOTE: (Kapsy) Another option I've read about is
                            // to flip the normal map normal, but that would
                            // look less natural, as there would be a jump from
                            // where the maximum valid reflection had been
                            // reached.
                            v34 reflplane = Unit (Cross (N1refl, V));
                            v34 surfplane = Unit (N0refl);
                            v34 clampedrefl = Unit (Cross (reflplane, surfplane));

                            // NOTE: (Kapsy) Obtain reflection color.
                            v34 refl = N1 + RandomInUnitSphere4Fast (seed);
                            m128 reflover = (Dot (refl, N0refl) < MM_ZERO);
                            v34 reflres = (refl & MM_INV (reflover)) + (clampedrefl & reflover);

                            ray4 rnew = Ray4 (p + N0refl*bias, reflres);
                            ApplyBonus (rnew, r4, mat);

                            colres4_t col1 = GetColorForRaySplittingBySign(&rnew, object, fastbsp, depth, seed);
                            col0.I = A*col1.I; // we only keep mixing in the color for recursive bounces, everything else is just oonc.e

                            r4->splitcount += rnew.splitcount;
                        }

                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;


            case MAT_DIELECTRIC:
                {
                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    colres4_t col0 = {};

                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);

                        v34 A = V34 (MM_ONE);//GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);
                        // do this in a way so we don't repeat everywhere.
                        // NEED TO MAKE SURE THAT THIS IS FOR THE FIRST HIT ONLY
                        // yep, only set albedo and a there, otherwise nothing at all
                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N1 + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }

                        // move this to options
                        m128 bias = _mm_set1_ps(1e-3);
                        v34 V = r4->dir;

                        m128 VdotN0 = Dot (V, N0);
                        m128 VdotN1 = Dot (V, N1);

                        v34 negV = V*MM_NEGONE;

                        m128 negVdotN0 = Dot (negV, N0);
                        m128 negVdotN1 = Dot (negV, N1);

                        // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                        m128 extmask = (negVdotN0 > MM_ZERO);
                        m128 intmask = MM_INV (extmask);

                        // NOTE: (Kapsy) Normal for reflected geometry rays.
                        v34 N0refl = (N0 & extmask) + (N0*MM_NEGONE & intmask);
                        // NOTE: (Kapsy) Normal for reflected normal map rays.
                        v34 N1refl = (N1 & extmask) + (N1*MM_NEGONE & intmask);

                        m128 negVdotN1refl = Dot (negV, N1refl);

                        v34 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                        // NOTE: (Kapsy) Obtain refraction color.
                        m128 refrindex = _mm_set1_ps(mat->refindex);
                        m128 lenVdotN1 = VdotN1/Length (r4->dir);

                        m128 cos =
                            _mm_and_ps(refrindex*lenVdotN1, intmask) +
                            _mm_and_ps(MM_NEGONE*lenVdotN1, extmask);

                        m128 niovernt =
                            _mm_and_ps(refrindex, intmask) +
                            _mm_and_ps(MM_ONE/refrindex, extmask);

                        v34 unitV = Unit (V);
                        m128 dt = Dot (unitV, N1refl);
                        m128 dtmask = (dt > MM_ZERO);
                        m128 discriminant = MM_ONE - niovernt*niovernt*(MM_ONE - dt*dt);
                        m128 discmask = (discriminant > MM_ZERO);

                        // NOTE: (Kapys) Obtain the reflection probability.
                        m128 reflfactor = _mm_set1_ps(mat->reflfactor);
                        m128 reflprob =
                            Clamp014 ((_mm_and_ps (Schlick4 (cos, refrindex), discmask) +
                                       _mm_and_ps (MM_ONE, MM_INV (discmask))) *reflfactor);
                        m128 reflmask = RandomUnilateral4 (seed) <= reflprob;


                        // NOTE: (Kapsy) Obtain refraction vector.
                        v34 refr = Unit (niovernt*(unitV - N1refl*dt) - N1refl*_mm_sqrt_ps (discriminant));
                        m128 refrover = (Dot (refr, N0refl) > MM_ZERO);
                        v34 refrres = (refr & MM_INV (refrover)) + (clampedrefl & refrover);

                        // NOTE: (Kapsy) Obtain reflection vector.
                        v34 refl = Unit (V + negVdotN1refl*MM_TWO*N1);
                        m128 reflover = (Dot (refl, N0refl) < MM_ZERO);
                        v34 reflres = (refl & MM_INV (reflover)) + (clampedrefl & reflover);

                        // NOTE: (Kapsy) Create new ray and get color.
                        v34 newbias = (N0refl*bias & reflmask) + (N0refl*MM_NEGONE*bias & MM_INV (reflmask));
                        v34 rnewdir = (reflres & reflmask) + (refrres & MM_INV (reflmask));
                        ray4 rnew = Ray4 (p + newbias, rnewdir);
                        ApplyBonus (rnew, r4, mat);

                        colres4_t col1 = GetColorForRaySplittingBySign(&rnew, object, fastbsp, depth, seed);
                        col0.I = A*col1.I; // we only keep mixing in the color for recursive bounces, everything else is just oonc.e

                        r4->splitcount += rnew.splitcount;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;


            case MAT_METAL:
                {
                    colres4_t col0 = {};

                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);

                        // move this to options
                        m128 bias = _mm_set1_ps(1e-3);

                        v34 A = GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);

                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N1 + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }


                        v34 V = r4->dir;

                        m128 VdotN0 = Dot (V, N0);
                        m128 VdotN1 = Dot (V, N1);

                        v34 negV = V*MM_NEGONE;

                        m128 negVdotN0 = Dot (negV, N0);
                        m128 negVdotN1 = Dot (negV, N1);

                        // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                        m128 extmask = (negVdotN0 > MM_ZERO);
                        m128 intmask = MM_INV (extmask);

                        // NOTE: (Kapsy) Normal for reflected geometry rays.
                        v34 N0refl = N0;
                        // NOTE: (Kapsy) Normal for reflected normal map rays.
                        v34 N1refl = N1;

                        m128 negVdotN1refl = Dot (negV, N1refl);

                        v34 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                        // NOTE: (Kapsy) Obtain reflection color.
                        v34 refl = Unit (V + negVdotN1refl*MM_TWO*N1);
                        m128 reflover = (Dot (refl, N0refl) < MM_ZERO);
                        v34 reflres = (refl & MM_INV (reflover)) + (clampedrefl & reflover);

                        ray4 rrefl = Ray4 (p + N0refl*bias, reflres + _mm_set1_ps (mat->fuzz)*RandomInUnitSphere4Fast (seed));
                        ApplyBonus (rrefl, r4, mat);

                        colres4_t col1 = GetColorForRaySplittingBySign(&rrefl, object, fastbsp, depth, seed);
                        col0.I = A*col1.I;

                        r4->splitcount += rrefl.splitcount;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;

            case MAT_METAL_DIR_COLOR:
                {
                    colres4_t col0 = {};

                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);

                        // move this to options
                        m128 bias = _mm_set1_ps(1e-3);

                        v34 A = V34 (_mm_set1_ps (0.6)) + ((r4->dir + V34 (MM_ONE))*V34 (MM_HALF))*V34 (_mm_set1_ps (0.4));

                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N1 + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }

                        v34 V = r4->dir;

                        m128 VdotN0 = Dot (V, N0);
                        m128 VdotN1 = Dot (V, N1);

                        v34 negV = V*MM_NEGONE;

                        m128 negVdotN0 = Dot (negV, N0);
                        m128 negVdotN1 = Dot (negV, N1);

                        // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                        m128 extmask = (negVdotN0 > MM_ZERO);
                        m128 intmask = MM_INV (extmask);

                        // NOTE: (Kapsy) Normal for reflected geometry rays.
                        v34 N0refl = N0;
                        // NOTE: (Kapsy) Normal for reflected normal map rays.
                        v34 N1refl = N1;

                        m128 negVdotN1refl = Dot (negV, N1refl);

                        v34 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                        // NOTE: (Kapsy) Obtain reflection color.
                        v34 refl = Unit (V + negVdotN1refl*MM_TWO*N1);
                        m128 reflover = (Dot (refl, N0refl) < MM_ZERO);
                        v34 reflres = (refl & MM_INV (reflover)) + (clampedrefl & reflover);

                        ray4 rrefl = Ray4 (p + N0refl*bias, reflres + _mm_set1_ps (mat->fuzz)*RandomInUnitSphere4Fast (seed));
                        ApplyBonus (rrefl, r4, mat);

                        colres4_t col1 = GetColorForRaySplittingBySign(&rrefl, object, fastbsp, depth, seed);
                        col0.I = A*col1.I;

                        r4->splitcount += rrefl.splitcount;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;

            case MAT_SOLID:
                {
                    colres4_t col0 = {};
                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);
                    if (r4->remdepth > 0)
                    {
                        v34 A = GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);
                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }

                        col0.I = A;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;

            case MAT_LAMBERTIAN_REFLECTION_MAP:
                {
                    m128 ks = _mm_set1_ps (0.3);

                    // should get earlier!
                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);
                    // should make a umatset, vmatset once so we don't have to keep doing so here.

                    colres4_t col0 = {};
                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);
                        // do this in a way so we don't repeat everywhere.

                        m128 bias = _mm_set1_ps(1e-3);

                        v34 A = GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);
                        // do this in a way so we don't repeat everywhere.
                        // NEED TO MAKE SURE THAT THIS IS FOR THE FIRST HIT ONLY
                        // yep, only set albedo and a there, otherwise nothing at all
                        //
                        m128 base = _mm_set1_ps (0.2f);
                        ks = base + A.x*(MM_ONE - base);
                        // NOTE: (Kapsy) Set the "dirt" albedo.
                        A = V34(V3(0.1, 0.1, 0.1));

                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N1 + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }

                        v34 V = r4->dir;

                        m128 VdotN0 = Dot (V, N0);
                        m128 VdotN1 = Dot (V, N1);

                        v34 negV = V*MM_NEGONE;

                        m128 negVdotN0 = Dot (negV, N0);
                        //m128 negVdotN1 = Dot (negV, N1);

                        // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                        // Don't need to do this for reflection only!
                        m128 extmask = (negVdotN0 > MM_ZERO);
                        m128 intmask = MM_INV (extmask);

                        // NOTE: (Kapsy) Normal for reflected geometry rays.
                        v34 N0refl = N0;//(N0 & extmask) + (N0*MM_NEGONE & intmask);
                        // NOTE: (Kapsy) Normal for reflected normal map rays.
                        v34 N1refl = N1;//(N1 & extmask) + (N1*MM_NEGONE & intmask);

                        m128 negVdotN1refl = Dot (negV, N1refl);

                        v34 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                        m128 reflprob = ReflectionProb (mat, V, N1refl, VdotN1, intmask, extmask);

                        // NOTE: (Kapsy) Obtain reflection vector.
                        v34 refl = Unit ((V + negVdotN1refl*MM_TWO*N1) + _mm_set1_ps (mat->fuzz)*RandomInUnitSphere4Fast (seed));
                        m128 reflover = (Dot (refl, N0refl) < MM_ZERO);
                        v34 reflres = (refl & MM_INV (reflover)) + (clampedrefl & reflover);

                        // NOTE: (Kapsy) Obtain Lambert scatter vector.
                        v34 lamb = N1 + RandomInUnitSphere4Fast (seed);
                        m128 lambover = (Dot (lamb, N0refl) < MM_ZERO);
                        v34 lambres = (lamb & MM_INV (lambover)) + (clampedrefl & lambover);

                        reflprob = reflprob*ks;
                        m128 reflmask = RandomUnilateral4 (seed) <= reflprob;

                        A = (V34 (MM_ONE) & reflmask) + (A & MM_INV (reflmask));

                        // NOTE: (Kapsy) Create new ray and get color.
                        v34 rnewdir = (reflres & reflmask) + (lambres & MM_INV (reflmask));
                        ray4 rnew = Ray4 (p + N0refl*bias, rnewdir);
                        ApplyBonus (rnew, r4, mat);

                        colres4_t col1 = GetColorForRaySplittingBySign(&rnew, object, fastbsp, depth, seed);
                        col0.I = A*col1.I;


                        r4->splitcount += rnew.splitcount;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;

            case MAT_DUMB_BRDF:
                {
                    // NOTE: (Kapsy) The ratio of reflection of the specular term of incoming light.
                    m128 ks = _mm_set1_ps (mat->Ks.r);
                    // NOTE: (Kapsy) The "shininess" of the specularity.
                    m128 ns = _mm_set1_ps (mat->Ns);

                    colres4_t col0 = {};

                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = GetTextureNormal (mat, p, N0, T, u, v, matsetmask);

                        v34 A = GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);

                        // TODO: (Kapsy) Do this in a way so we don't repeat everywhere.
                        // NEED TO MAKE SURE THAT THIS IS FOR THE FIRST HIT ONLY.
                        // Yep, only set albedo and a there, otherwise nothing at all.
                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N1 + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }

                        col0.A = A;

                        m128 bias = _mm_set1_ps(1e-3);

                        v34 V = r4->dir;

                        m128 VdotN0 = Dot (V, N0);
                        m128 VdotN1 = Dot (V, N1);

                        v34 negV = V*MM_NEGONE;

                        m128 negVdotN0 = Dot (negV, N0);

                        // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                        // Don't need to do this for reflection only!
                        m128 extmask = (negVdotN0 > MM_ZERO);
                        m128 intmask = MM_INV (extmask);

                        // rename to out???
                        // NOTE: (Kapsy) Normal for reflected geometry rays.
                        v34 N0refl = N0;
                        // NOTE: (Kapsy) Normal for reflected normal map rays.
                        v34 N1refl = N1;

                        m128 negVdotN1refl = Dot (negV, N1refl);

                        v34 clampedrefl = ClampedReflection (N0refl, N1refl, V);

                        // NOTE: (Kapsy) Obtain reflection vector.
                        v34 refl = Unit (V + negVdotN1refl*MM_TWO*N1 + ns*RandomInUnitSphere4Fast (seed));
                        m128 reflover = (Dot (refl, N0refl) < MM_ZERO);
                        v34 reflres = (refl & MM_INV (reflover)) + (clampedrefl & reflover);

                        // NOTE: (Kapsy) Obtain Lambert scatter vector.
                        v34 lamb = N1 + RandomInUnitSphere4Fast (seed);
                        m128 lambover = (Dot (lamb, N0refl) < MM_ZERO);
                        v34 lambres = (lamb & MM_INV (lambover)) + (clampedrefl & lambover);

                        m128 reflprob = ks;
                        m128 reflmask = RandomUnilateral4 (seed) <= reflprob;

                        A = (V34 (MM_ONE) & reflmask) + (A & MM_INV (reflmask));

                        // NOTE: (Kapsy) Create new ray and get color.
                        v34 rnewdir = (reflres & reflmask) + (lambres & MM_INV (reflmask));
                        ray4 rnew = Ray4 (p + N0refl*bias, rnewdir);
                        ApplyBonus (rnew, r4, mat);

                        colres4_t col1 = GetColorForRaySplittingBySign(&rnew, object, fastbsp, depth, seed);
                        col0.I = A*col1.I;

                        r4->splitcount += rnew.splitcount;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;

            case MAT_CAR_PAINT:
                {
                    colres4_t col0 = {};

                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    if (r4->remdepth > 0)
                    {
                        v34 N0 = N;
                        v34 N1 = N;
                        v34 A = GetAttenuation4 (mat->tex, _mm_and_ps (u, matsetmask), _mm_and_ps (v, matsetmask), p);

                        // TODO: (Kapsy) Do this in a way so we don't repeat everywhere.
                        // NEED TO MAKE SURE THAT THIS IS FOR THE FIRST HIT ONLY.
                        // Yep, only set albedo and a there, otherwise nothing at all.
                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = A;
                            col0.N = (((N + V34 (MM_ONE))*V34 (MM_HALF)) & matsetmask);
                        }
                        if (r4->remdepth == mat->remdepth)
                        {
                            col0.A = V34 (_mm_set1_ps (0.1f));
                        }

                        m128 bias = _mm_set1_ps(1e-3);

                        v34 V = r4->dir;
                        m128 VdotN0 = Dot (V, N0);
                        m128 VdotN1 = Dot (V, N1);

                        v34 negV = V*MM_NEGONE;

                        m128 negVdotN0 = Dot (negV, N0);

                        // NOTE: (Kapsy) Find which side of the actual geometry the ray is on.
                        // Don't need to do this for reflection only!
                        m128 extmask = (negVdotN0 > MM_ZERO);
                        m128 intmask = MM_INV (extmask);

                        // NOTE: (Kapsy) Normal for reflected geometry rays.
                        v34 N0refl = N0;
                        // NOTE: (Kapsy) Normal for reflected normal map rays.
                        v34 N1refl = N1;

                        m128 negVdotN1refl = Dot (negV, N1refl);

                        //v34 clampedrefl = ClampedReflection (N0refl, N1refl, V);
                        m128 reflprob = ReflectionProb (mat, V, N1refl, VdotN1, intmask, extmask);

                        // NOTE: (Kapsy) Obtain reflection vector.
                        m128 ns = _mm_set1_ps (mat->Ns);
                        v34 refl = Unit (V + negVdotN1refl*MM_TWO*N1) + ns*RandomInUnitSphere4Fast(seed);
                        v34 reflres = refl;

                        // NOTE: (Kapsy) Obtain Lambert scatter vector.
                        v34 lamb = N1 + RandomInUnitSphere4Fast (seed);
                        v34 lambres = lamb;

                        m128 reflmask = RandomUnilateral4 (seed) <= reflprob;

                        // shouldn't A1 be what we use for denoise?
                        // no
                        A = (V34 (MM_ONE) & reflmask) + (A & MM_INV (reflmask));

                        v34 rnewdir = (reflres & reflmask) + (lambres & MM_INV (reflmask));
                        ray4 rnew = Ray4 (p + N0refl*bias, rnewdir);
                        ApplyBonus (rnew, r4, mat);

                        // Needs to fill out I and A.
                        // Can we just make these functions return a colres? much easier.
                        // Okay either way is fine as long as we define the rules:
                        // - Each color result function takes a pointer
                        // - Internally, it creates a stack res
                        // - Does what it needs to do on that
                        // - Then it += writes that to the pointer
                        // The problem here is we have multiple colors per material.
                        //
                        // Still like the return way better. Stack based return just suits recursive processing better.
                        // Okay fine I have this all wrong.
                        // albedo is FIRST HIT ONLY
                        // same with NORMAL
                        // So we don't keep mixing it.
                        colres4_t col1 = GetColorForRaySplittingBySign(&rnew, object, fastbsp, depth, seed);
                        col0.I = A*col1.I; // We only keep mixing in the color for recursive bounces, everything else is just once.

                        r4->splitcount += rnew.splitcount;
                    }

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;


            case MAT_BACKGROUND:
                {
                    colres4_t col0 = {};

                    m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    // NOTE: (Kapsy) Not filling out N for sky.
                    col0.I = GetBackgroundColor (r4, mat);
                    col0.A = col0.I;
                    col0.N = V34 (MM_ZERO);

                    res.I += (col0.I & matsetmask);
                    res.A += (col0.A & matsetmask);
                    res.N += (col0.N & matsetmask);

                } break;

            case MAT_NORMALS:
                {
                    //// v34 I = V34 (V3(0,0,0));

                    //// m128 matsetmask = _mm_loadu_ps ((float *) matset->mask);

                    //// if (r4->remdepth > 0)
                    //// {
                    ////     I = ((N + V34 (MM_ONE))* V34 (MM_HALF));
                    //// }

                    //// col4[COLOR_INDEX] += (I & matsetmask);

                } break;

            case MAT_WUV:
            case MAT_SCATTER:
                {
                    Assert ("We do not handle these types.");
                } break;
        }
#endif
    }


    // NOTE: (Kapsy) Outmask should no longer apply if only tracing full packets.
    // The one reason I can think why our packet traces are much slower is simply because there are many paths were we break down to scalar...
    //// col4[COLOR_INDEX] = col4[COLOR_INDEX] & outmaskinv;
    //// col4[ALBEDO_INDEX] = col4[ALBEDO_INDEX] & outmaskinv;
    //// col4[NORMAL_INDEX] = col4[NORMAL_INDEX] & outmaskinv;

    res.I = res.I & outmaskinv;
    res.A = res.A & outmaskinv;
    res.N = res.N & outmaskinv;

    //// res4.x = _mm_and_ps (res4.x, outmaskinv);
    //// res4.y = _mm_and_ps (res4.y, outmaskinv);
    //// res4.z = _mm_and_ps (res4.z, outmaskinv);

    return (res);
}

static colres4_t
GetColorForRaySplittingBySign (ray4 *r4, object_t *object, fastbsp_t *fastbsp, int depth, randomseed_t *seed)
{
    colres4_t res = {};

#if 0
    res.I = V34 (MM_ONE);
    res.A = V34 (MM_ONE);
    // TODO: (Kapsy) THIS IS WRONG!
    res.N = V34 (MM_ONE); // don't use this.

#else
    v34 illum = V34 (MM_ZERO);
    for (int i=0 ; i<SIMD_WIDTH ; i++)
    {
          if (
              ((r4->A.x[i]) != (r4->A.x[i])) ||
              ((r4->A.y[i]) != (r4->A.y[i])) ||
              ((r4->A.z[i]) != (r4->A.z[i])) ||
              ((r4->B.x[i]) != (r4->B.x[i])) ||
              ((r4->B.y[i]) != (r4->B.y[i])) ||
              ((r4->B.z[i]) != (r4->B.z[i]))
             )
          {
              continue;
          }

        ray r = {};
        r.A.x = r4->A.x[i];
        r.A.y = r4->A.y[i];
        r.A.z = r4->A.z[i];
        r.B.x = r4->B.x[i];
        r.B.y = r4->B.y[i];
        r.B.z = r4->B.z[i];

        r.remdepth = r4->remdepth;
        r.havebonus = r4->havebonus;

        colres_t out0 = GetColorForRay (&r, object, fastbsp, seed);

        res.I.x[i] += out0.I.x;
        res.I.y[i] += out0.I.y;
        res.I.z[i] += out0.I.z;

        res.A.x[i] += out0.A.x;
        res.A.y[i] += out0.A.y;
        res.A.z[i] += out0.A.z;

        res.N.x[i] += out0.N.x;
        res.N.y[i] += out0.N.y;
        res.N.z[i] += out0.N.z;

        //// res.A[i] += out0.A;
        //// res.N[i] += out0.N;

        ////             //// res.I += (col0.I & matsetmask);
        ////             //// res.A += (col0.A & matsetmask);
        ////             //// res.N += (col0.N & matsetmask);

        //// illum.x[i] += res.x;
        //// illum.y[i] += res.y;
        //// illum.z[i] += res.z;
    }

#endif

#if 0
    // Really need a way of pulling this logic out to deal with color and shadows with the same code.
    // NOTE: (Kapsy) Split up the rays based on direction signs.

    m128 zero = _mm_set1_ps (0.f);
    //m128 all = _mm_set1_epi32 (0xffffffff);
    m128 processed = zero;

    m128 posx = r4->dir.x >= zero;
    m128 posy = r4->dir.y >= zero;
    m128 posz = r4->dir.z >= zero;

    m128 negx = r4->dir.x < zero;
    m128 negy = r4->dir.y < zero;
    m128 negz = r4->dir.z < zero;

    {
        // sure we could make a much faster sign check using bit masks.
    m128 ppp = And3 (posx, posy, posz);
    if (HaveBitsSet (ppp))
    {
        v3 signs1 = V3 (0,0,0);
        // TODO: (Kapsy) Really need to take into account the material mask here too.

        // Getting a shadow/reflection bug which means I think we're not choosing our single rays properly.
        // Might just be the sky.
        // need to do the logic here better, for instance, if == 1 else if > 1
        // Getting a moire pattern that is only caused with single rays. Almost certain it is due to not handling the metal shader.
        //if (BitsSetCount (ppp) == 1)
        if (0)
        {
            int i = BitsSetIndex (ppp);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(ppp), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, ppp);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 nnn = And3 (negx, negy, negz);;
    if (HaveBitsSet (nnn))
    {
        v3 signs1 = V3 (1,1,1);
        //if (BitsSetCount (nnn) == 1)
        if (0)
        {
            int i = BitsSetIndex (nnn);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(nnn), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, nnn);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 pnn = And3 (posx, negy, negz);
    if (HaveBitsSet (pnn))
    {
        v3 signs1 = V3 (0,1,1);
        //if (BitsSetCount (pnn) == 1)
        if (0)
        {
            int i = BitsSetIndex (pnn);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(pnn), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, pnn);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 npp = And3 (negx, posy, posz);
    if (HaveBitsSet (npp))
    {
        v3 signs1 = V3 (1,0,0);
        //if (BitsSetCount (npp) == 1)
        if (0)
        {
            int i = BitsSetIndex (npp);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(npp), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, npp);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 pnp = And3 (posx, negy, posz);
    if (HaveBitsSet (pnp))
    {
        v3 signs1 = V3 (0,1,0);
        //if (BitsSetCount (pnp) == 1)
        if (0)
        {
            int i = BitsSetIndex (pnp);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(pnp), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, pnp);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 npn = And3 (negx, posy, negz);
    if (HaveBitsSet (npn))
    {
        v3 signs1 = V3 (1,0,1);
        //if (BitsSetCount (npn) == 1)
        if (0)
        {
            int i = BitsSetIndex (npn);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(npn), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, npn);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 ppn = And3 (posx, posy, negz);
    if (HaveBitsSet (ppn))
    {
        v3 signs1 = V3 (0,0,1);

        //if (BitsSetCount (ppn) == 1)
        if (0)
        {
            int i = BitsSetIndex (ppn);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(ppn), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, ppn);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }

    m128 nnp = And3 (negx, negy, posz);
    if (HaveBitsSet (nnp))
    {
        v3 signs1 = V3 (1,1,0);
        //if (BitsSetCount (nnp) == 1)
        if (0)
        {
            int i = BitsSetIndex (nnp);
            ray r = {};
            r.A.x = r4->A.x[i];
            r.A.y = r4->A.y[i];
            r.A.z = r4->A.z[i];
            r.B.x = r4->B.x[i];
            r.B.y = r4->B.y[i];
            r.B.z = r4->B.z[i];

            r.remdepth = r4->remdepth;
            r.havebonus = 0;

            v3 res = GetColorForRay (&r, object, fastbsp, seed);
            illum.x[i] += res.x;
            illum.y[i] += res.y;
            illum.z[i] += res.z;
        }
        else
        {
            illum += GetColorForRay4 (r4, object, fastbsp, (depth + 1), Inv(nnp), signs1, seed);
        }
        r4->splitcount++;
        OrE (processed, nnp);
        if (AllBitsSet4 (processed))
            goto finishlambert;
    }
    }
#endif

finishlambert:

    return (res);
}












#define CAMERA_TEST 0

#define SINGLE_FRAME_TEST 0

#if 0
static void
CreateDebugBoundingBox (rect3 aabb)
{
    v3 p = aabb.min;
    v3 q = aabb.max;

    object *object = bbobjects + bbobjectcount++;

    int meshindex = object->meshcount++;
    object->meshes = (mesh *)malloc(sizeof(mesh)*object->meshcount);
    mesh *mesh = object->meshes + meshindex;

    mesh->aabb = aabb;
    mesh->tricount = 12;
    mesh->tris = (tri *)malloc(mesh->tricount*sizeof(tri));

    texture *bbtex = &textures[texturecount++];
    bbtex->type = TEX_PLAIN;
    bbtex->albedo = V3 (1,0,0);
    mesh->mat = (material) { MAT_WUV, 0 };

    tri *triat = mesh->tris;

    // front
    triat->A = V3 (p.x, p.y, q.z);
    triat->B = V3 (q.x, p.y, q.z);
    triat->C = V3 (q.x, q.y, q.z);
    triat++;
    triat->A = V3 (q.x, q.y, q.z);
    triat->B = V3 (p.x, q.y, q.z);
    triat->C = V3 (p.x, p.y, q.z);
    triat++;

    // right
    triat->A = V3 (q.x, p.y, q.z);
    triat->B = V3 (q.x, p.y, p.z);
    triat->C = V3 (q.x, q.y, p.z);
    triat++;
    triat->A = V3 (q.x, q.y, p.z);
    triat->B = V3 (q.x, q.y, q.z);
    triat->C = V3 (q.x, p.y, q.z);
    triat++;

    // back
    triat->A = V3 (q.x, p.y, p.z);
    triat->B = V3 (p.x, p.y, p.z);
    triat->C = V3 (p.x, q.y, p.z);
    triat++;
    triat->A = V3 (p.x, q.y, p.z);
    triat->B = V3 (q.x, q.y, p.z);
    triat->C = V3 (q.x, p.y, p.z);
    triat++;

    // left
    triat->A = V3 (p.x, p.y, p.z);
    triat->B = V3 (p.x, p.y, q.z);
    triat->C = V3 (p.x, q.y, q.z);
    triat++;
    triat->A = V3 (p.x, q.y, q.z);
    triat->B = V3 (p.x, q.y, p.z);
    triat->C = V3 (p.x, p.y, p.z);
    triat++;

    // top
    triat->A = V3 (p.x, q.y, q.z);
    triat->B = V3 (q.x, q.y, q.z);
    triat->C = V3 (q.x, q.y, p.z);
    triat++;
    triat->A = V3 (q.x, q.y, p.z);
    triat->B = V3 (p.x, q.y, p.z);
    triat->C = V3 (p.x, q.y, q.z);
    triat++;

    // bottom
    triat->A = V3 (p.x, p.y, p.z);
    triat->B = V3 (q.x, p.y, p.z);
    triat->C = V3 (q.x, p.y, q.z);
    triat++;
    triat->A = V3 (q.x, p.y, q.z);
    triat->B = V3 (p.x, p.y, q.z);
    triat->C = V3 (p.x, p.y, p.z);
    triat++;

}
#endif

enum
{
    JobTypeRenderChunk,
    JobTypeToneMap,
};

struct jobqueueentry_t
{
    int type;
    void *data;
};

#define MAX_JOB_QUEUE_ENTRIES (1 << 14)
struct jobqueue_t
{
    int completioncount;
    int completiongoal;
    dispatch_semaphore_t semaphorehandle;
    unsigned int readpos;
    unsigned int writepos;
    jobqueueentry_t entries[MAX_JOB_QUEUE_ENTRIES];
};

static jobqueue_t g_jobqueue;

static void
AddJobToQueue (jobqueue_t *queue, int type, void *job)
{
    // NOTE: (Kapsy) This function should only be called from the same thread, so no reason to atomically set writepos.
    jobqueueentry_t *entry = queue->entries + queue->writepos;
    entry->type = type;
    entry->data = job;

    int newwritepos = (queue->writepos + 1) & (MAX_JOB_QUEUE_ENTRIES - 1);
    Assert (newwritepos != queue->readpos);
    queue->writepos = newwritepos;

    ++queue->completiongoal;

    __atomic_thread_fence(0);

    dispatch_semaphore_signal(queue->semaphorehandle);
}

struct renderchunkdata_t
{
    pixelbuffer_t *buffers;
    object_t *object;
    camera *cam; // 24

    int dimx;
    int dimy;
    int starty;
    int startx;

    int nx;
    int ny; // 48

    int clearbuffer; // 52

    // could make ints to save more room and multiply in render threads.
    float stratposx;
    float stratposy; // 60
    int rppstart;

    randomseed_t randomseed;

    // NOTE: (Kapsy) Interestingly, getting this wrong causes a 2-3 % slowdown.
    // char pad[4];
};


static void RenderChunk (renderchunkdata_t *data);
static void ToneMap (renderchunkdata_t *data);

void *
ThreadProc (void *arg)
{
    jobqueue_t *queue = &g_jobqueue;

    for (;;)
    {
        unsigned int readpos = queue->readpos;
        if (readpos != queue->writepos)
        {
            // NOTE: (Kapsy) Doesn't really need to be a circular buffer at this stage.
            unsigned int newreadpos = (readpos + 1) & (MAX_JOB_QUEUE_ENTRIES - 1);
            if (__sync_bool_compare_and_swap (&queue->readpos, readpos, newreadpos))
            {
                // TODO: (Kapsy) Job entries should also be spaced a cacheline apart.
                jobqueueentry_t *entry = queue->entries + readpos;

                switch (entry->type)
                {
                    case JobTypeRenderChunk:
                        {
                            renderchunkdata_t *data = (renderchunkdata_t *)entry->data;
                            RenderChunk (data);

                        } break;

                    case JobTypeToneMap:
                        {
                            renderchunkdata_t *data = (renderchunkdata_t *)entry->data;
                            ToneMap (data);

                        } break;
                }

                __sync_fetch_and_add (&queue->completioncount, 1);
            }
        }
        else
        {
            dispatch_semaphore_wait (queue->semaphorehandle, DISPATCH_TIME_FOREVER);
        }
    }
}


static void
CompleteAllJobs (void)
{
    jobqueue_t *queue = &g_jobqueue;

    // TODO: (Kapsy) Make THIS thread do some work.
    // NOTE: (Kapsy) Assuming that the completion goal doesn't change here.
    unsigned int completioncount = __sync_fetch_and_add (&queue->completioncount, 0);
    while (completioncount != queue->completiongoal)
    {
        // Move to inline function.
        unsigned int readpos = queue->readpos;
        if (readpos != queue->writepos)
        {
            // NOTE: (Kapsy) Doesn't really need to be a circular buffer at this stage.
            unsigned int newreadpos = (readpos + 1) & (MAX_JOB_QUEUE_ENTRIES - 1);
            if (__sync_bool_compare_and_swap (&queue->readpos, readpos, newreadpos))
            {
                // TODO: (Kapsy) Job entries should also be spaced a cacheline apart.
                jobqueueentry_t *entry = queue->entries + readpos;

                switch (entry->type)
                {
                    case JobTypeRenderChunk:
                        {
                            renderchunkdata_t *data = (renderchunkdata_t *)entry->data;
                            RenderChunk (data);

                        } break;

                    case JobTypeToneMap:
                        {
                            renderchunkdata_t *data = (renderchunkdata_t *)entry->data;
                            ToneMap (data);

                        } break;
                }

                __sync_fetch_and_add (&queue->completioncount, 1);
            }
        }

        completioncount = __sync_fetch_and_add (&queue->completioncount, 0);
    }
}

static int g_renderchunkcount;
static renderchunkdata_t *g_renderchunks;

#define ELEMENTS_PER_PIXEL 3

#define SIMD_PACKET_1X4 0
// NOTE: (Kapsy) Seems a bit faster. Need to profile.
#define SIMD_PACKET_2X2 1

static void
RenderChunk (renderchunkdata_t *chunk)
{
    int rppstart = chunk->rppstart;

    randomseed_t *seed = &chunk->randomseed;

    // NOTE: (Kapsy) For debugging individual packets only.
    // NOTE: (Kapsy) testx MUST BE EVEN, testy MUST BE ODD.
    // int testx = 112;
    // int testy = 73;

    int testx = 314;
    int testy = 581;

    Assert ((chunk->dimx & SIMD_WIDTH) == 0);
    Assert ((chunk->dimy & SIMD_WIDTH) == 0);

    int nx = chunk->nx;
    int ny = chunk->ny;

    camera *cam = chunk->cam;

    object_t *object = chunk->object;
    fastbsp_t *fastbsp = g_fastbsp; // should tie to object

    int startx = chunk->startx;
    int endx = chunk->startx + chunk->dimx; // should always be a multiple of simd w

    int starty = chunk->starty;
    int endy = chunk->starty + chunk->dimy; // should always be a multiple of simd h

    // NOTE: (Kapsy) Even though we say aa, this is actually more MC.
    // rename to rpp
    int aawcount = 1;
    int aahcount = 1;
    int aacount = aawcount*aahcount;
    float aacountinv = 1.f/(float)aacount;

#if SIMD_PACKET_2X2

    int packetw = 2;
    int packeth = 2;

    unsigned int w = chunk->buffers[COLOR_INDEX].w;
    unsigned int h = chunk->buffers[COLOR_INDEX].h;

    float stratposx = chunk->stratposx;
    float stratposy = chunk->stratposy;
    // float lensstratangle = chunk->lensstratangle;

    // NOTE: (Kapsy) 2X2 packet process.
    for (int j=(endy - 1) ; j >= starty ; j-=packeth)
    {
        for (int i=startx ; i < endx; i+=packetw)
        {
            ray4 r4 = {};
            r4.remdepth = MAX_DEPTH;
            r4.havebonus = 1;

            colres4_t col0 = {};

            if ((j == testy) && (i == testx))
            {
                int breakhere = 12345;
            }

            // NOTE: (Kapsy) Have confirmed that there is no performance
            // difference between an aa count of 1, and no aa loop at all.
            // However, one major disadvantage to this method is packet
            // coverage is larger, meaning there is more of a chance for the
            // individual rays to collide with different prims.
            // For MC would like to investigate packets/pixel < 1 way of doing things.
            // Should make quite a big difference for shaders especially.
            // What we have now is more suitable for real time, however for
            // real time we should be approaching anit aliasing in a different
            // way anyway.
            // for (int a=0 ; a<aacount ; a++)
            // float awd = 1.f/(float)aawcount;
            // float ahd = 1.f/(float)aahcount;

            for (int aw=0 ; aw<aawcount ; aw++)
            {
                for (int ah=0 ; ah<aahcount ; ah++)
                {
                    __sync_fetch_and_add (&numprimaryrays, 4);

                    // NOTE: (Kapsy) Have to define rays in packet from bottom to top.
                    // Can probably SIMD this.
                    for (int y=(packeth - 1) ; y >= 0 ; y--)
                    {
                        for (int x=0 ; x<packetw ; x++)
                        {
                            // NOTE: (Kapsy) Stratified samples.
                            // This definitely improves the "togetherness" of aa.
                            // float randx = ((float)aw)*awd + drand48()*awd;
                            // float randy = ((float)ah)*ahd + drand48()*ahd;

                            // float randx = (float)stratposx + drand48()*g_stratdimx;
                            // float randy = (float)stratposy + drand48()*g_stratdimy;

                            float randx = drand48();
                            float randy = drand48();

                            float u = ((float)(i + x) + randx)/(float)nx;
                            float v = ((float)(j - y) + randy)/(float)ny; // TODO: (Kapsy) Should this be minus rand?

                            // float u = (float)(i + x)/(float)nx;
                            // float v = (float)(j - y)/(float)ny;

                            int sy = h - (j - y + 1);
                            int sx = i;
                            int a = sy*nx*g_rppcount + sx*g_rppcount + rppstart;
                            float lensstratangle = g_stratangles[a];

                            int k = y*packetw + x;

                            ray r1 = GetRay (cam, u, v, lensstratangle);

                            // NOTE: (Kapsy) All single rays test.
                            //// r1.remdepth = MAX_DEPTH;
                            //// r1.havebonus = 1;
                            //// v3 rescol = GetColorForRay (&r1, object, fastbsp, seed);

                            //// col4.x[k] += rescol.x;
                            //// col4.y[k] += rescol.y;
                            //// col4.z[k] += rescol.z;
#if 1
                            r4.orig.x[k] = r1.orig.x;
                            r4.orig.y[k] = r1.orig.y;
                            r4.orig.z[k] = r1.orig.z;

                            r4.dir.x[k] = r1.dir.x;
                            r4.dir.y[k] = r1.dir.y;
                            r4.dir.z[k] = r1.dir.z;
#endif 
                        }
                    }

                    m128 zero = _mm_set1_ps (0.f);
                    //m128 all = _mm_set1_epi32 (0xffffffff);
                    m128 processed = zero;

                    m128 posx = r4.dir.x >= zero;
                    m128 posy = r4.dir.y >= zero;
                    m128 posz = r4.dir.z >= zero;

                    m128 negx = r4.dir.x < zero;
                    m128 negy = r4.dir.y < zero;
                    m128 negz = r4.dir.z < zero;

                    m128 ppp = And3 (posx, posy, posz);
                    m128 nnn = And3 (negx, negy, negz);;
                    m128 pnn = And3 (posx, negy, negz);
                    m128 npp = And3 (negx, posy, posz);
                    m128 pnp = And3 (posx, negy, posz);
                    m128 npn = And3 (negx, posy, negz);
                    m128 ppn = And3 (posx, posy, negz);
                    m128 nnp = And3 (negx, negy, posz);

                    colres4_t col1 = {};

                    if (AllBitsSet4 (ppp))
                    {
                        v3 signs1 = V3 (0,0,0);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(ppp), signs1, seed);
                    }

                    else if (AllBitsSet4 (nnn))
                    {
                        v3 signs1 = V3 (1,1,1);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(nnn), signs1, seed);
                    }

                    else if (AllBitsSet4 (pnn))
                    {
                        v3 signs1 = V3 (0,1,1);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(pnn), signs1, seed);
                    }

                    else if (AllBitsSet4 (npp))
                    {
                        v3 signs1 = V3 (1,0,0);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(npp), signs1, seed);
                    }

                    else if (AllBitsSet4 (pnp))
                    {
                        v3 signs1 = V3 (0,1,0);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(pnp), signs1, seed);
                    }

                    else if (AllBitsSet4 (npn))
                    {
                        v3 signs1 = V3 (1,0,1);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(npn), signs1, seed);
                    }

                    else if (AllBitsSet4 (ppn))
                    {
                        v3 signs1 = V3 (0,0,1);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(ppn), signs1, seed);
                    }

                    else if (AllBitsSet4 (nnp))
                    {
                        v3 signs1 = V3 (1,1,0);
                        col1 = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(nnp), signs1, seed);
                    }
                    else
                    {
                        for (int i=0 ; i<SIMD_WIDTH ; i++)
                        {
                            //// if (
                            ////         ((r4.A.x[i]) != (r4.A.x[i])) ||
                            ////         ((r4.A.y[i]) != (r4.A.y[i])) ||
                            ////         ((r4.A.z[i]) != (r4.A.z[i])) ||
                            ////         ((r4.B.x[i]) != (r4.B.x[i])) ||
                            ////         ((r4.B.y[i]) != (r4.B.y[i])) ||
                            ////         ((r4.B.z[i]) != (r4.B.z[i]))
                            ////    )
                            //// {
                            ////     continue;
                            //// }

                            ray r = {};
                            r.A.x = r4.A.x[i];
                            r.A.y = r4.A.y[i];
                            r.A.z = r4.A.z[i];
                            r.B.x = r4.B.x[i];
                            r.B.y = r4.B.y[i];
                            r.B.z = r4.B.z[i];

                            r.remdepth = r4.remdepth;
                            r.havebonus = r4.havebonus;

                            colres_t col2 = GetColorForRay (&r, object, fastbsp, seed);

                            col1.I.x[i] += col2.I.x;
                            col1.I.y[i] += col2.I.y;
                            col1.I.z[i] += col2.I.z;

                            col1.A.x[i] += col2.A.x;
                            col1.A.y[i] += col2.A.y;
                            col1.A.z[i] += col2.A.z;

                            col1.N.x[i] += col2.N.x;
                            col1.N.y[i] += col2.N.y;
                            col1.N.z[i] += col2.N.z;
                        }
                    }

#if 0
                    //col4 = ((r4.dir + V34 (MM_ONE))* V34 (MM_HALF));

                    // NOTE: (Kapsy) All packets method of splitting rays.
                    if ((j == testy) && (i == testx))
                    {
                        int breakhere = 1234;
                    }

                    // NOTE: (Kapsy) Split up the rays based on direction signs.

                    m128 zero = _mm_set1_ps (0.f);
                    //m128 all = _mm_set1_epi32 (0xffffffff);
                    m128 processed = zero;

                    m128 posx = r4.dir.x >= zero;
                    m128 posy = r4.dir.y >= zero;
                    m128 posz = r4.dir.z >= zero;

                    m128 negx = r4.dir.x < zero;
                    m128 negy = r4.dir.y < zero;
                    m128 negz = r4.dir.z < zero;

                    m128 ppp = And3 (posx, posy, posz);
                    if (HaveBitsSet (ppp))
                    {
                        v3 signs1 = V3 (0,0,0);
                        col4 += GetColorForRay4 (&r4, object, fastbsp, 0, Inv(ppp), signs1, seed);
                        r4.splitcount++;
                        OrE (processed, ppp);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 nnn = And3 (negx, negy, negz);;
                    if (HaveBitsSet (nnn))
                    {
                        v3 signs1 = V3 (1,1,1);
                        col4 += GetColorForRay4 (&r4, object, fastbsp, 0, Inv(nnn), signs1, seed);
                        r4.splitcount++;
                        OrE (processed, nnn);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 pnn = And3 (posx, negy, negz);
                    if (HaveBitsSet (pnn))
                    {
                        v3 signs1 = V3 (0,1,1);
                        col4 += GetColorForRay4 (&r4, object, fastbsp, 0, Inv(pnn), signs1, seed);
                        r4.splitcount++;
                        OrE (processed, pnn);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 npp = And3 (negx, posy, posz);
                    if (HaveBitsSet (npp))
                    {
                        v3 signs1 = V3 (1,0,0);
                        col4 += GetColorForRay4 (&r4, object, fastbsp, 0, Inv(npp), signs1, seed);
                        r4.splitcount++;
                        OrE (processed, npp);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 pnp = And3 (posx, negy, posz);
                    if (HaveBitsSet (pnp))
                    {
                        v3 signs1 = V3 (0,1,0);
                        col4 += GetColorForRay4 (&r4, object, fastbsp, 0, Inv(pnp), signs1, seed);
                        r4.splitcount++;
                        OrE (processed, pnp);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 npn = And3 (negx, posy, negz);
                    if (HaveBitsSet (npn))
                    {
                        v3 signs1 = V3 (1,0,1);
                        v34 colres = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(npn), signs1, seed);
                        r4.splitcount++;
                        col4 += colres;
                        OrE (processed, npn);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 ppn = And3 (posx, posy, negz);
                    if (HaveBitsSet (ppn))
                    {
                        v3 signs1 = V3 (0,0,1);
                        v34 colres = GetColorForRay4 (&r4, object, fastbsp, 0, Inv(ppn), signs1, seed);
                        r4.splitcount++;
                        col4 += colres;
                        OrE (processed, ppn);
                        if (AllBitsSet4 (processed))
                            continue;
                    }

                    m128 nnp = And3 (negx, negy, posz);
                    if (HaveBitsSet (nnp))
                    {
                        v3 signs1 = V3 (1,1,0);
                        col4 += GetColorForRay4 (&r4, object, fastbsp, 0, Inv(nnp), signs1, seed);
                        r4.splitcount++;
                        OrE (processed, nnp);
                        if (AllBitsSet4 (processed))
                            continue;
                    }
#endif

                    // NOTE: (Kapsy) Post stuff here.
                    col0.I += col1.I;
                    col0.A += col1.A;
                    col0.N += col1.N;
                }
            }

            for (int y=0 ; y<packeth ; y++)
            {
                unsigned int w = chunk->buffers[COLOR_INDEX].w;
                unsigned int h = chunk->buffers[COLOR_INDEX].h;

                unsigned int bufferoffset =
                    (h - (j - y + 1))*w*ELEMENTS_PER_PIXEL +
                    (i)*ELEMENTS_PER_PIXEL;

                //// float *bufferat = chunk->buffer->data +
                ////     (chunk->buffer->h - (j - y + 1))*chunk->buffer->w*ELEMENTS_PER_PIXEL +
                ////     (i)*ELEMENTS_PER_PIXEL;

                float *colorat = chunk->buffers[COLOR_INDEX].data + bufferoffset;
                float *albedoat = chunk->buffers[ALBEDO_INDEX].data + bufferoffset;
                float *normalat = chunk->buffers[NORMAL_INDEX].data + bufferoffset;

                for (int x=0 ; x<packetw ; x++)
                {
                    int k = y*packetw + x;

                    {
                        v3 col = {};
                        col.r = col0.I.x[k]*g_rppinv;
                        col.g = col0.I.y[k]*g_rppinv;
                        col.b = col0.I.z[k]*g_rppinv;

                        if (chunk->clearbuffer)
                        {
                            *(colorat++) = col.r;
                            *(colorat++) = col.g;
                            *(colorat++) = col.b;
                        }
                        else
                        {
                            *(colorat++) = (*colorat) + col.r;
                            *(colorat++) = (*colorat) + col.g;
                            *(colorat++) = (*colorat) + col.b;
                        }
                    }
                    {
                        v3 col = {};
                        col.r = col0.A.x[k]*g_rppinv;
                        col.g = col0.A.y[k]*g_rppinv;
                        col.b = col0.A.z[k]*g_rppinv;

                        if (chunk->clearbuffer)
                        {
                            *(albedoat++) = col.r;
                            *(albedoat++) = col.g;
                            *(albedoat++) = col.b;
                        }
                        else
                        {
                            *(albedoat++) = (*albedoat) + col.r;
                            *(albedoat++) = (*albedoat) + col.g;
                            *(albedoat++) = (*albedoat) + col.b;
                        }
                    }
                    {
                        v3 col = {};
                        col.r = col0.N.x[k]*g_rppinv;
                        col.g = col0.N.y[k]*g_rppinv;
                        col.b = col0.N.z[k]*g_rppinv;

                        if (chunk->clearbuffer)
                        {
                            *(normalat++) = col.r;
                            *(normalat++) = col.g;
                            *(normalat++) = col.b;
                        }
                        else
                        {
                            *(normalat++) = (*normalat) + col.r;
                            *(normalat++) = (*normalat) + col.g;
                            *(normalat++) = (*normalat) + col.b;
                        }
                    }
                }
            }
        }
    }
#endif

    //// printf("RenderChunk finished.\n");
    //// fflush (0);
}

static void
ToneMap (renderchunkdata_t *chunk)
{
    Assert ((chunk->dimx & SIMD_WIDTH) == 0);
    Assert ((chunk->dimy & SIMD_WIDTH) == 0);

    int nx = chunk->nx;
    int ny = chunk->ny;

    int startx = chunk->startx;
    int endx = chunk->startx + chunk->dimx; // Should always be a multiple of simd w

    int starty = chunk->starty;
    int endy = chunk->starty + chunk->dimy; // Should always be a multiple of simd h

    int packetw = 2;
    int packeth = 2;

    for (int j=(endy - 1) ; j >= starty ; j-=packeth)
    {
        for (int i=startx ; i < endx; i+=packetw)
        {
            for (int y=0 ; y<packeth ; y++)
            {
                unsigned int w = chunk->buffers[COLOR_INDEX].w;
                unsigned int h = chunk->buffers[COLOR_INDEX].h;

                unsigned int bufferoffset =
                    (h - (j - y + 1))*w*ELEMENTS_PER_PIXEL +
                    (i)*ELEMENTS_PER_PIXEL;

                float *colorat = chunk->buffers[COLOR_INDEX].data + bufferoffset;
                float *albedoat = chunk->buffers[ALBEDO_INDEX].data + bufferoffset;
                float *normalat = chunk->buffers[NORMAL_INDEX].data + bufferoffset;
                float *outat = chunk->buffers[OUTPUT_INDEX].data + bufferoffset;

                // TODO: (Kapsy) Need to use SIMD here.
                for (int x=0 ; x<packetw ; x++)
                {
                    int k = y*packetw + x;

                    v3 col = {};
                    col.x = Clamp01 (colorat[0]);
                    col.y = Clamp01 (colorat[1]);
                    col.z = Clamp01 (colorat[2]);

                    //float shape = 1.08f;
                    //float shape = 1.12f;
                    float shape = 1.2f;
                    col.x = sqrt (powf (col.x, shape));
                    col.y = sqrt (powf (col.y, shape));
                    col.z = sqrt (powf (col.z, shape));

                    *colorat++ = col.r;
                    *colorat++ = col.g;
                    *colorat++ = col.b;
                }
            }
        }
    }

    for (int j=(endy - 1) ; j >= starty ; j-=packeth)
    {
        for (int i=startx ; i < endx; i+=packetw)
        {
            for (int y=0 ; y<packeth ; y++)
            {
                unsigned int w = chunk->buffers[COLOR_INDEX].w;
                unsigned int h = chunk->buffers[COLOR_INDEX].h;

                unsigned int bufferoffset =
                    (h - (j - y + 1))*w*ELEMENTS_PER_PIXEL +
                    (i)*ELEMENTS_PER_PIXEL;

                float *colorat = chunk->buffers[COLOR_INDEX].data + bufferoffset;
                float *albedoat = chunk->buffers[ALBEDO_INDEX].data + bufferoffset;
                float *normalat = chunk->buffers[NORMAL_INDEX].data + bufferoffset;
                float *outat = chunk->buffers[OUTPUT_INDEX].data + bufferoffset;

                // TODO: (Kapsy) Need to use SIMD here.
                for (int x=0 ; x<packetw ; x++)
                {
                    int k = y*packetw + x;

                    v3 col = {};
                    col.x = Clamp01 (outat[0]);
                    col.y = Clamp01 (outat[1]);
                    col.z = Clamp01 (outat[2]);

                    float shape = 1.2f;
                    col.x = sqrt (powf (col.x, shape));
                    col.y = sqrt (powf (col.y, shape));
                    col.z = sqrt (powf (col.z, shape));

                    *outat++ = col.r;
                    *outat++ = col.g;
                    *outat++ = col.b;
                }
            }
        }
    }
}


#define MAX_VNTS (1 << 12)

// NOTE: (Kapsy) A temp list for storing all tangents per vertex normal.
struct vntlist_t
{
    unsigned int count;
    v3 tangents[MAX_VNTS];
};

// TODO: (Kapsy) Need to put all this into a scene struct.
// TODO: (Kapsy) Just passsing all this junk for now, but will eventually merge it into some kind of camera settings struct.
static void
SimulateScene (float tdelta, camera &cam, object_t *object, fastbsp_t *fastbsp, int nx, int ny,
        v3 &lookfrom, v3 &lookat,
        float &focusdist, float &targetfocusdist,
        v3 vup, float halfwidth, float halfheight, float rotspeed)
{

    // Shouldn't need to do this if we have a simple ray test for the pixel?
    cam.origin = lookfrom;
    cam.w = Unit (lookfrom - lookat);
    cam.u = Unit (Cross (vup, cam.w));
    cam.v = Cross (cam.w, cam.u);

    cam.lowerleft = cam.origin - halfwidth*focusdist*cam.u - halfheight*focusdist*cam.v - focusdist*cam.w;
    cam.horiz = 2.f*halfwidth*focusdist*cam.u;
    cam.vert = 2.f*halfheight*focusdist*cam.v;

    // TODO: (Kapsy) Need to make this function take the u, v, so we can setup the camera after.
    targetfocusdist = GetAutofocusDistance (&cam, object, fastbsp, nx, ny);

    // NOTE: (Kapsy) Can't set an update rate here?
    // Yes we can, just has be be t based, instead of frame or ticks.
    // Don't want this to be constant.
    // Although, might actually suit being constant.
    // So that big focus changes would take longer to kick in.
    // So need it to be so that if no more changes to target. Trying fixed for now.

    float focusdistdiff = targetfocusdist - focusdist;

    // NOTE: (Kapsy) In seconds.
    float focussmoothingtime = 0.24;
    float focusupdaterate = focusdistdiff/focussmoothingtime;//(in units/s)

    // TODO: (Kapsy) Ideally would get the delta from this t and last t, so we can jump to allow for shutter speed.
    float focusdistdelta = tdelta*focusupdaterate;
    focusdist += focusdistdelta;

    if ((focusupdaterate > 0.f && focusdist >= targetfocusdist) ||
        (focusupdaterate < 0.f && focusdist <= targetfocusdist))
    {
        focusupdaterate = 0.f;
        focusdist = targetfocusdist;
    }

    // NOTE: (Kapsy) Recalc camera bounds with new focus dist.
    cam.lowerleft = cam.origin - halfwidth*focusdist*cam.u -
                    halfheight*focusdist*cam.v - focusdist*cam.w;

    cam.horiz = 2.f*halfwidth*focusdist*cam.u;
    cam.vert = 2.f*halfheight*focusdist*cam.v;

#if 1
    // Rodrigues Rotation formula
    v3 v = lookfrom;
    v3 k = V3 (0,1,0);
    float omega = rotspeed*tdelta;
    v = v*cos(omega) + Cross (k, v)*sin(omega) + k*Dot (k, v)*(1.f - cos(omega));
    lookfrom = v;
#endif

}

static unsigned int g_seedfactor = 1;

int main(int argc, char **argv)
{
    printf ("Started\n");
    fflush (0);

    // NOTE: (Kapsy) Setup point lights.
    pointlight_t *l1 = g_pointlights + g_pointlightscount++;
    l1->col = V3 (1,0.5,1);
    l1->col = V3 (1);
    l1->p = V3 (-2.9, 0.6, -0.0);
    l1->intensity = 24.0f;

    pointlight_t *l2 = g_pointlights + g_pointlightscount++;
    l2->col = V3 (0.1,0.9,1);
    l2->p = V3 (-0.7, 0.9, -0.0);
    l2->intensity = 11.0f;

    pointlight_t *l3 = g_pointlights + g_pointlightscount++;
    l3->col = V3 (0.2,0.2,1);
    l3->p = V3 (0.7, 0.9, -1.3);
    l3->intensity = 18.0f;

    // NOTE: (Kapsy) Setup a simple memory pool.
    uint64_t g_bspmempoolsize = Megabytes ((1024 + 512));
    g_bspmempool.base = (char *) malloc (g_bspmempoolsize);
    g_bspmempool.at = g_bspmempool.base;
    g_bspmempool.remaining = g_bspmempoolsize;

#if 1
    // m44 T = M44Trans (0.f, 0.f, 0.f);
    // object_t *object = LoadOBJObject ("data/stanford_bunny_01.obj", T);

    m44 T = M44Trans (0.f, 0.0f, -0.3f);
    object_t *object = LoadOBJObject ("data/merc_300sl_45_internal_paint_reflections_fix.obj", T);

    // m44 T = M44Trans (0.f, 0.f, 0.f);
    // object_t *object = LoadOBJObject ("data/test_cube_02.obj", T);

    // m44 T = M44Trans (0.f, 0.f, 0.f);
    // object_t *object = LoadOBJObject ("data/low_poly_test_001.obj", T);

    // NOTE: (Kapsy) Manually creating tangents for each vertex:
    // Iterate all triangles:
    // Create T
    // For each vertex used by the triangle, we create a T vector, mapped to verts size for now.
    // We also create a list for that t, and add the triangles T to that list.
    // Go through each t, average out the ts from the tri t list.

    // For now, don't need special tri_t for the t indexes themselves, they just map to tri/vert indexes. Wasteful, but just to get it going should be fine.

    // Okay, something major we failed to take into account was that we are
    // averaging ALL tangents when we only want to be averaging those with
    // shared vertex normals.
    //
    // So what we should do is build our temp list of tangents per vertex
    // normal, instead of per vertex.
    //
    // And we would retrive them in the same way, so that there may be shared
    // tangents.

    // TODO: (Kapsy) Move this out to the obj loader.
    vntlist_t *vntlists = (vntlist_t *) calloc (1, sizeof (vntlist_t)*object->vertnormcount);

    for (int i=0 ; i<object->tricount ; i++)
    {
        tri_t *tri = object->tris + i;

        v3 A = object->verts[tri->A];
        v3 B = object->verts[tri->B];
        v3 C = object->verts[tri->C];

        int matindex = object->trimats[i];
        mat_t *mat = object->mats + matindex;
        if (mat->texnorm)
        {
            tri_t *trivt = object->trivts + i;
            Assert (trivt->A >= 0);
            Assert (trivt->B >= 0);
            Assert (trivt->C >= 0);

            v3 uvA = object->vertuvs[trivt->A];
            v3 uvB = object->vertuvs[trivt->B];
            v3 uvC = object->vertuvs[trivt->C];

            // TODO: (Kapsy) Check on these!
            v3 e1 = B - A;
            v3 e2 = C - A;

            v3 deluv1 = uvB - uvA;
            v3 deluv2 = uvC - uvA;

            float f = 1.f/(deluv1.u*deluv2.v - deluv2.u*deluv1.v);

            // NOTE: (Kapsy) Obtain the t value for the flat triangle
            v3 t = V3 (0.f);

            t.x = f*(deluv2.v*e1.x - deluv1.v*e2.x);
            t.y = f*(deluv2.v*e1.y - deluv1.v*e2.y);
            t.z = f*(deluv2.v*e1.z - deluv1.v*e2.z);

            tri_t *trivn = object->trivns + i;

            for (int j=0 ; j<3 ; j++)
            {
                // NOTE: (Kapsy) Mapping to vertex normals for now.
                // Add all t values to list to be interpolated.
                vntlist_t *vntlist = vntlists + trivn->e[j];
                Assert (vntlist->count < MAX_VNTS);
                v3 *trit = vntlist->tangents + vntlist->count++;
                *trit = t;
            }
        }
    }

    object->tangents = (v3 *)malloc(sizeof(v3)*object->vertnormcount);

    // NOTE: (Kapsy) Interpolate all tangents per vertex, only for those with a valid list.
    for (int i=0 ; i<object->vertnormcount ; i++)
    {
        vntlist_t *vntlist = vntlists + i;

        if (vntlist->count)
        {
            v3 accumt = V3 (0.f);

            for (int j=0 ; j<vntlist->count ; j++)
            {
                accumt += vntlist->tangents[j];
            }

            float denom = 1.f/(float)vntlist->count;
            accumt = Unit (accumt*V3 (denom));

            AssertNAN (accumt.x);
            AssertNAN (accumt.y);
            AssertNAN (accumt.z);

            object->tangents[i] = accumt;
        }
    }

    free(vntlists);

    v3 lookfrom = V3 (0.0,0.0,7.3);
    v3 lookat = V3 (0.f,1.2f,0.0);

    // NOTE: (Kapsy) Temp settings only.
#if 0
    for (int i=0 ; i<object->matcount ; i++)
    {
        mat_t *mat = object->mats + i;
        mat->type = MAT_LAMBERTIAN;
        mat->depthbonus = 0;
        mat->tex->albedo = V3(1.f);

        //// mat->type = MAT_METAL;
        //// mat->fuzz = 0.0f;

        //// switch (i)
        //// {
        ////         // chrome
        ////     case 0:
        ////         {

        ////             mat->type = MAT_METAL;
        ////             mat->fuzz = 0.0f;
        ////             //
        ////             //mat->type = MAT_LAMBERTIAN;
        ////             //mat->tex->albedo = V3 (1,1,0.5);
        ////             /////mat->tex->type = TEX_PLAIN;

        ////         } break;

        ////         // glass edge
        ////     case 1:
        ////         {

        ////         } break;

        ////         // bump
        ////     case 2:
        ////         {
        ////             mat->type = MAT_DIELECTRIC;
        ////             mat->refindex = 1.1f;

        ////         } break;

        ////         // smooth
        ////     case 3:
        ////         {
        ////             mat->type = MAT_DIELECTRIC;
        ////             mat->refindex = 1.1f;

        ////         } break;

        //// }


    }
#endif

#if 1
    // NOTE: (Kapsy) Post process set the proper material type.
    // Will eventually switch on original types, or create base on properties.
    for (int i=0 ; i<object->matcount ; i++)
    {
        mat_t *mat = object->mats + i;
        mat->type = MAT_LAMBERTIAN;
        mat->depthbonus = 0;

        switch (i)
        {

            case MercMatBlackTrim:
                {
                } break;

            case MercMatBlackTrim2:
                {
                } break;

            case MercMatBody:
                {
                    // NOTE: (Kapsy) Original black paint.
#if 0
                    mat->tex->albedo = V3(116.f/255.f, 119.f/255.f, 102.f/255.f)*0.0;
                    mat->type = MAT_CAR_PAINT;

                    mat->ksbase = 0.33f;
                    mat->ksmax = 1.f;

                    mat->Ns = 0.010f;
                    mat->refindex = 1.3f;

                    mat->depthbonus = 2*BONUS_MOD;
                    mat->reflfactor = 3.f;
#endif

                    // NOTE: (Kapsy) Hot wheels.
#if 0
                    mat->tex->albedo = V3(1.f);
                    mat->type = MAT_METAL_DIR_COLOR;
                    mat->fuzz = 0.010f;

                    // TODO: (Kapsy) Should split out for wheels and maybe bumpers onl?
                    mat->depthbonus = 4*BONUS_MOD;
#endif

                    // NOTE: (Kapsy) Red paint.
#if 1
                    mat->tex->albedo = V3(255.f/255.f, 0.f/255.f, 0.f/255.f);
                    mat->type = MAT_CAR_PAINT;

                    mat->Ns = 0.000f;
                    mat->refindex = 3.4f;

                    mat->depthbonus = 2*BONUS_MOD;
                    mat->reflfactor = 1.0f;
#endif
                } break;

            case MercMatChrome:
            case MercMatChromeTrim:
                {
#if 1
                    mat->tex->albedo = V3(1.f);
                    mat->type = MAT_METAL;
                    mat->fuzz = 0.010f;
                    mat->depthbonus = 4*BONUS_MOD;
#endif

                } break;

            case MercMatDarkChrome:
                {
                } break;

            case MercMatGauges:
                {
                } break;

            case MercMatGlass:
                {
                    mat->type = MAT_DIELECTRIC;
                    mat->refindex = 2.7f;
                    mat->reflfactor = 1.1f;
                    mat->depthbonus = 2;
                } break;

            case MercMatHeadlampLensFlat:
                {
                    mat->type = MAT_DIELECTRIC;
                    mat->refindex = 1.3f;
                    mat->reflfactor = 2.4f;
                    mat->depthbonus = 6*BONUS_MOD;

                } break;

            case MercMatHeadlampBulb:
            case MercMatHeadlampLensBump:
                {
                    mat->type = MAT_DIELECTRIC;
                    mat->refindex = 1.3f;
                    mat->reflfactor = 2.4f;
                    mat->depthbonus = 0*BONUS_MOD;

                } break;

            case MercMatLights:
                {
                    mat->type = MAT_DUMB_BRDF;
                    mat->Kd = V3 (0.6f);
                    mat->Ks = V3 (0.4f);
                    mat->Ns = 0.05f;

                } break;

            case MercMatLogo:
                {
                } break;

            case MercMatMirror:
                {
                    mat->tex->albedo = V3(1.f);
                    mat->type = MAT_METAL;
                    mat->fuzz = 0.0f;

                } break;

            case MercMatRed_Carpet:
                {
                    mat->tex->albedo = V3(228.f/255.f, 197.f/255.f, 128.f/255.f)*1.1;
                    mat->type = MAT_LAMBERTIAN;

                } break;

            case MercMatRedLeather:
            case MercMatRedLeather2:
                {
                    mat->tex->albedo = V3(228.f/255.f, 197.f/255.f, 128.f/255.f)*1.1;

                    mat->type = MAT_DUMB_BRDF;
                    mat->Kd = V3 (0.95f);
                    mat->Ks = V3 (0.05f);
                    mat->Ns = 0.35f;
                    mat->texnorm = 0;

                } break;

            case MercMatRubber:
            case MercMatRubberTrim:
                {
                    mat->type = MAT_DUMB_BRDF;
                    mat->tex->albedo = mat->tex->albedo*0.8f;
                    mat->Kd = V3 (0.8f);
                    mat->Ks = V3 (0.2f);
                    mat->Ns = 0.3f;

                } break;

            case MercMatWhiteTrim:
                {
                    mat->type = MAT_DUMB_BRDF;
                    mat->Kd = V3 (0.6f);
                    mat->Ks = V3 (0.4f);
                    mat->Ns = 0.1f;

                } break;

            case MercMatWinkerGlass:
                {
                    mat->type = MAT_LAMBERTIAN;
                    mat->tex->albedo = V3(1.3);
                } break;

            case MercMatBathtub:
                {
                    mat->tex->type = TEX_PLAIN;
                    mat->type = MAT_SOLID;
                    mat->tex->albedo = V3(0);

                } break;


        }

        // TODO: (Kapsy) A bit dumb, should add the bonus, but we only apply the bonus _after_ the first hit.
        mat->remdepth = MAX_DEPTH;
    }
#endif

    // TODO: (Kapsy) Can't think of a better way to do this yet, but will eventually move all materials to the scene rather than the object.
    InitPerlin(&testperlin2, 0.00003f);
    texture *cloudtexture = &textures[texturecount++];
    cloudtexture->type = TEX_PERLIN2;
    cloudtexture->perlin = &testperlin2;
    //cloudtexture->albedo = V3 (140.f/255.f, 110.f/255.f, 125.f/255.f);
    cloudtexture->albedo = V3 (1.f); // make this the reflection map

    // TODO: (Kapsy) Can't think of a better way to do this yet, but will eventually move all materials to the scene rather than the object.
    object->bgmatindex = object->matcount++;
    object->mats = (mat_t *)realloc((void *)object->mats, sizeof(mat_t)*object->matcount);
    mat_t *bgmat = object->mats + object->bgmatindex;
    bgmat->type = MAT_BACKGROUND;
    bgmat->tex = cloudtexture;

#elif 0

    // test texture cube
    m44 T = M44Trans (0.f, 0.0f, 3.0f);
    object_t *object = LoadOBJObject ("data/test_tex_cube_03.obj", T);

    v3 lookfrom = V3 (0.0, 0.0, 7.3);
    v3 lookat = V3 (0.f, 1.2f, 0.0);

    for (int i=0 ; i<object->matcount ; i++)
    {
        mat_t *mat = object->mats + i;
        mat->type = MAT_LAMBERTIAN;

        switch (i)
        {
            case 0:
                {
                mat->tex->albedo = V3(1.0);
                mat->type = MAT_DUMB_BRDF;
                mat->Kd = V3 (0.6f);
                mat->Ks = V3 (0.4f);
                mat->Ns = 0.009f;
                } break;
        }
    }

    InitPerlin(&testperlin2, 0.00003f);
    texture *cloudtexture = &textures[texturecount++];
    cloudtexture->type = TEX_PERLIN2;
    cloudtexture->perlin = &testperlin2;
    cloudtexture->albedo = V3 (1.f); // make this the reflection map

    // TODO: (Kapsy) Can't think of a better way to do this yet, but will eventually move all materials to the scene rather than the object.
    object->bgmatindex = object->matcount++;
    object->mats = (mat_t *)realloc((void *)object->mats, sizeof(mat_t)*object->matcount);
    mat_t *bgmat = object->mats + object->bgmatindex;
    bgmat->type = MAT_BACKGROUND;
    bgmat->tex = cloudtexture;

#endif

    if (!LoadFastBSP (g_fastbsp))
    {

          //////////////////////////////////////////////////////////////////////
         //// Setup Make BSP Tree /////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        printf ("Setup Make BSP Tree\n");
        fflush (0);

        numtris = object->tricount;

        // NOTE: (Kapsy) Use system alloc for temp nodes here.
        g_makenodes = (makenode_t *) malloc( sizeof (makenode_t)*MAX_BSP_NODES);
        makenode_t *makenode = g_makenodes + g_makenodecount++;

        makenode->type = NodeTypeInner;
        makenode->split = 0.f;
        makenode->leftchild = 0;
        makenode->rightchild = 0;
        makenode->V = object->aabb;

        makenode->trilist.count = object->tricount;
        makenode->trilist.indexes =  (int *)malloc(sizeof(int)*makenode->trilist.count);

        for (int i=0 ; i<object->tricount ; i++ )
        {
            makenode->trilist.indexes[i] = i;
        }

        printf ("Checking MakeBSPNode input...\n");

        printf ("t:%d s:%f\n", makenode->type, makenode->split);

        printf ("V min:%.03f %.03f %.03f max:%.03f %.03f %.03f\n",
                makenode->V.min.x,
                makenode->V.min.y,
                makenode->V.min.z,
                makenode->V.max.x,
                makenode->V.max.y,
                makenode->V.max.z);

        printf ("makenode->trilist.count:%d\n", makenode->trilist.count);

        MakeBSPNode(object, makenode, 0);

        printf ("Leaf creation stats:\n  Hit min tri count: %d\n  Hit max BSP depth: %d\n  Best cost found: %d\n", g_mintricount, g_maxbspdepth, g_costbestfound);
        fflush (0);


          //////////////////////////////////////////////////////////////////////
         //// Make BSP Tree Stats /////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        g_makenodepairs = (makenodepair_t *)malloc(sizeof(makenodepair_t)*MAX_BSP_DEPTH);
        g_makenodepaircount = 0;

        CollateSlowBSPDebugInfo (makenode);

        // MakeManualSplitTest(mesh->tris, mesh->tricount, makenode, 0);

          //////////////////////////////////////////////////////////////////////
         //// Setup Fast BSP Tree /////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        printf ("Setup Fast BSP Tree\n");
        fflush (0);

        // Stack used only for creation?
        g_makenodepaircount = 0;

        // Need sizes for all allocations? Not really, as long as we have the count we should know how much to read from the file.
        g_fastbsp = (fastbsp_t *) PoolAlloc (&g_bspmempool, sizeof (*g_fastbsp), CACHELINE_SIZE);

        // NOTE: (Kapsy) In order to align node pairs to cacheline boundaries,
        // the root node must start at + 1.
        // Ensure tri accs are pool alloc'd after the fast nodes (for offsets).

        // To serialize this would need all these globals in a file.
        // Could put them in a struct with some sizes, wouldn't be hard to resurrect from that.
        // Then here, we just check the existence of a file, and create if not, otherwise read in.
        // Need to make sure we alloc mem aligned when loading.
        bsp_node *nodes = (bsp_node *) PoolAlloc (&g_bspmempool, sizeof (bsp_node)*MAX_BSP_NODES, CACHELINE_SIZE);
        P32AssignP (g_fastbsp->nodes, nodes);
        g_fastbsp->nodecount = 1;

        bsp_node *fastnode = P32ToP (g_fastbsp->nodes, bsp_node) + g_fastbsp->nodecount++;
        SetBSPType(fastnode, NodeTypeInner);
        SetBSPDim(fastnode, makenode->dim);

        // NOTE: (Kapsy) Allocate a list of accelaration structures.
        // To keep memory offsets positive, this must be allocated after the nodes.
        // TODO: (Kapsy) Should have an accurate triangle count here!
        triacclist_t *triacclist = (triacclist_t *) PoolAlloc (&g_bspmempool, sizeof (triacclist_t)*MAX_BSP_NODES, CACHELINE_SIZE);
        P32AssignP (g_fastbsp->triacclist, triacclist);
        g_fastbsp->triacclistcount = 0;

        // NOTE: (Kapsy) Allocate twice as many to account for overlap.
        // Really need to go through a simple ex
        // this is obviously wrong... because we are including tris that touch our boundaries...
        g_fastbsp->triacccount = g_bspstat_NTT;// Pobject->tricount*40;
        triacc *triaccsbase = (triacc *) PoolAlloc (&g_bspmempool, sizeof (triacc)*g_fastbsp->triacccount, CACHELINE_SIZE);
        P32AssignP (g_fastbsp->triaccsbase, triaccsbase);

        // g_fastbsp->triaccat = g_fastbsp->triaccsbase;
        CreateFastBSPFromSlow (g_fastbsp, object, makenode, fastnode);

        // DebugPrintFastBSP(&g_fastbsp, 128);
        free (g_makenodes);
        free (g_makenodepairs);

        printf ("Create fast BSP complete!\n");

        SaveFastBSP (g_fastbsp);

        printf ("Save fast BSP complete!\n");

        fflush (0);

    }

#if 0
      //////////////////////////////////////////////////////////////////////////
     //// Fast BSP Tree Stats /////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    //what do I want to know?
    //for each leaf, how many tris do we traverse
    //how deep is each leaf?

    bsp_node *node = g_fastnodes + 1;

    int bspstackcount = 0;
    stack_item bspstack[MAX_BSP_STACK_COUNT];

    unsigned int totaltris = 0;
    unsigned int totalleaves = 0;

    for(;;)
    {
        while (!BSP_IsLeaf(node))
        {
            {
                float backdir = 0.f;
                float frontdir = 0.f;

                // Case 3: Traverse both sides in turn
                bsp_node *backchild = GetBackChild (node, backdir);
                PushStack (bspstack, bspstackcount, backchild, 0.f, 0.f);

                printf ("Fast BSP inner\n");
                node = GetFrontChild (node, frontdir);
            }
        }

        // NOTE: (Kapsy) We now have a leaf.
        // Print leaf stats, tri count and depth
        int offset = BSP_Offset(node);
        triacclist_t *list = (triacclist_t *)((char *)node + offset);
        printf("Fast BSP leaf tris: %d depth: %d\n", list->count, bspstackcount);
        fflush (0);

        totaltris+=list->count;
        totalleaves++;

        if (bspstackcount == 0)
            break; // nothing left over to traverse

        stack_item *stack = PopStack (bspstack, bspstackcount);
        node = stack->node;
    }

        printf("Fast total leaves: %d tris: %d\n", totalleaves, totaltris);
        fflush (0);
#endif

#if CAMERA_TEST

    // NOTE: (Kapsy) Camera test spheres.
    // NOTE: (Kapsy) 45 deg radius should fill 90 deg (vert) fov.
    float R = cos(M_PI/4.f);

    //// sphere *s0 = &spheres[spherecount++];
    //// s0->center = V3 (-R, 0.f, -1.f);
    //// s0->rad = R;
    //// s0->mat = (material) { MAT_LAMBERTIAN, V3 (0.f, 0.f, 1.f) };

    //// sphere *s1 = &spheres[spherecount++];
    //// s1->center = V3 (R, 0.f, -1.f);
    //// s1->rad = R;
    //// s1->mat = (material) { MAT_LAMBERTIAN, V3 (1.f, 0.f, 0.f) };

#endif

#if 0
    texture *tex = &textures[texturecount++];
    tex->type = TEX_CHECKER; // not the best way to do this but...
    //tex->perlin = &testperlin;
    tex->albedo = V3 (1.f);

    int smat1index = object->matcount++;
    // TODO: (Kapsy) Can't think of a better way to do this yet, but will eventually move all materials to the scene rather than the object.
    object->mats = (mat_t *)realloc((void *)object->mats, sizeof(mat_t)*object->matcount);
    mat_t *smat1 = object->mats + smat1index;
    smat1->type = MAT_SOLID;
    smat1->texnorm = 0;
    smat1->tex = tex;

    sphere *s1 = &spheres[spherecount++];
    s1->center = V3 (0.f, -2000.23f, 0.f); // for lowered
    s1->rad = 2000.f;
    s1->matindex = smat1index;
#endif

#if 0
    InitPerlin (&testperlin, 1.f);

    // inefficient, because we use the same noise for both...
    testperlin.scale = 10.f;
    texture *ntex = &textures[texturecount++];
    ntex->type = TEX_PERLIN_NORMAL; // might have to make a perlin noise type...
    ntex->perlin = &testperlin;
    ntex->albedo = V3 (1.f);

    texture *tex = &textures[texturecount++];
    tex->type = TEX_PERLIN; // not the best way to do this but...
    tex->perlin = &testperlin;
    tex->albedo = V3 (1.f);

    int smat1index = object->matcount++;
    // TODO: (Kapsy) Can't think of a better way to do this yet, but will eventually move all materials to the scene rather than the object.
    object->mats = (mat_t *)realloc((void *)object->mats, sizeof(mat_t)*object->matcount);
    mat_t *smat1 = object->mats + smat1index;
    smat1->type = MAT_LAMBERTIAN_REFLECTION_MAP;
    smat1->depthbonus = 0*BONUS_MOD;

    smat1->texnorm = ntex;
    smat1->tex = tex;
    smat1->refindex = 1.f;
    smat1->reflfactor = 1.7f;

    sphere *s1 = &spheres[spherecount++];
    s1->center = V3 (0.f, -2000.23f, 0.f); // for lowered
    s1->rad = 2000.f;
    s1->matindex = smat1index;
#endif


#if 1
    InitPerlin (&testperlin, 1.f);
    testperlin.scale = 10.f;

    texture *normtexsrc = &textures[texturecount++];
    normtexsrc->type = TEX_PERLIN_NORMAL;
    normtexsrc->perlin = &testperlin;
    normtexsrc->albedo = V3 (1.f);

    texture *masktexsrc = &textures[texturecount++];
    masktexsrc->type = TEX_CHECKER;
    masktexsrc->perlin = &testperlin;
    masktexsrc->albedo = V3 (1.f);

    int groundmatindex = object->matcount++;
    // TODO: (Kapsy) Can't think of a better way to do this yet, but will eventually move all materials to the scene rather than the object.
    object->mats = (mat_t *)realloc((void *)object->mats, sizeof(mat_t)*object->matcount);
    mat_t *groundmat = object->mats + groundmatindex;
    groundmat->type = MAT_LAMBERTIAN_REFLECTION_MAP;
    groundmat->depthbonus = 3*BONUS_MOD;
    groundmat->remdepth = MAX_DEPTH;

    groundmat->texnorm = 0;
    groundmat->tex = masktexsrc;
    groundmat->refindex = 1.f;
    groundmat->reflfactor = 1.7f;
    groundmat->fuzz = 0.2f;

    sphere *groundsphere = &spheres[spherecount++];
    groundsphere->center = V3 (0.f, -2000.23f, 0.f); // for lowered
    groundsphere->rad = 2000.f;
    groundsphere->matindex = groundmatindex;
    //groundsphere->coordtype = CoordType_YProjection;

#if 0
      //////////////////////////////////////////////////////////////////////////
     //// Baking Perlin Noise /////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    unsigned int nxt = (1 << 11);
    unsigned int nzt = nxt;

    texture *normtexdest = &textures[texturecount++];
    normtexdest->type = TEX_NORMAL;
    normtexdest->albedo = V3 (1.f);
    texbuf_t *normbuf = &normtexdest->bufa;
    normbuf->w = nxt;
    normbuf->h = nzt;
    normbuf->cpp = 3;
    normbuf->e = (unsigned char *)malloc (sizeof (normbuf->e[0])*normbuf->w*normbuf->h*normbuf->cpp);

    texture *masktexdest = &textures[texturecount++];
    masktexdest->type = TEX_BITMAP;
    masktexdest->albedo = V3 (1.f);
    texbuf_t *maskbuf = &masktexdest->bufa;
    maskbuf->w = nxt;
    maskbuf->h = nzt;
    maskbuf->cpp = 3;
    maskbuf->e = (unsigned char *)malloc (sizeof (maskbuf->e[0])*maskbuf->w*maskbuf->h*maskbuf->cpp);

    float dim = groundsphere->rad*2;

    float halfdim = dim*0.5f;
    float xmin = -halfdim;
    float xmax = halfdim;

    float zmin = -halfdim;
    float zmax = halfdim;

    unsigned char *normbufat = normbuf->e;
    unsigned char *maskbufat = maskbuf->e;

    // should really simd this.
    for (int z=0 ; z<nzt ; z++)
    {
        for (int x=0 ; x<nxt ; x++)
        {
            float rayx = ((float)x/(float)nxt)*dim - halfdim;
            float rayz = ((float)z/(float)nzt)*dim - halfdim;

            if ((z == 512) && (x == 512))
            {
                int a = 1234;

                int b = 1234;

            }

            ray r = Ray (V3 (rayx, 1.f, rayz), V3 (0.f, -1.f, 0.f));
            hitrec hit = {};
            hit.dist = MAXFLOAT;

            // should just trav the one.
            TraverseSpheres (&r, &hit);

            v3 mask = V3 (0.f);
            v3 norm = V3 (0.f);



            if (hit.dist < MAXFLOAT)
            {
                // find the point of collision
                int sphereindex = (int)hit.primref;
                sphere *s = spheres + sphereindex;

                //float rad = s->rad;
                //v3 center = s->center;

                v3 p = r.orig + (hit.dist)*r.dir;
                //v3 N = (p - center)/rad;

                mat_t *mat = object->mats + s->matindex;
                mask = GetAttenuation (mat->tex, 0, 0, p);
                norm = GetAttenuation (mat->texnorm, 0, 0, p);
            }

            *maskbufat = (unsigned char)(mask.z*255.99);
            maskbufat++;
            *maskbufat = (unsigned char)(mask.z*255.99);
            maskbufat++;
            *maskbufat = (unsigned char)(mask.z*255.99);
            maskbufat++;

            //// *normbufat = (unsigned char)(norm.x*255.99);
            //// normbufat++;
            //// *normbufat = (unsigned char)(norm.y*255.99);
            //// normbufat++;
            //// *normbufat = (unsigned char)(norm.z*255.99);
            //// normbufat++;

            //// *maskbufat = (unsigned char)(1.f*255.99);
            //// maskbufat++;
            //// *maskbufat = (unsigned char)(0.f*255.99);
            //// maskbufat++;
            //// *maskbufat = (unsigned char)(0.f*255.99);
            //// maskbufat++;

        }
    }

    groundmat->texnorm = 0;//normtexdest;
    groundmat->tex = masktexdest;

#endif


#endif

      //////////////////////////////////////////////////////////////////////////
     //// Animation Setup /////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    float framerate = 60.f;
    // float framerate = 30.f;
    // float framerate = 15.f;
    // float framerate = 1.f;
    // float framerate = 12.f/9.f;

    float frametime = 1.f/framerate;
    float shuttertime = frametime;
    // float shuttertime = 0.f;

    float durations = 9.f;
    float framecount = framerate*durations;
    float rotspeed = (2.f*M_PI)/durations;
    // float ellipsephase = 0;

#if SINGLE_FRAME_TEST
    framecount = 1;
#endif

    // Retina
    // 2880/1800 1.6:1

    // int nx = 2880;
    // int ny = 1800;

    // HD
    // int nx = 1920;
    // int ny = 1080;

    // Close to HD
    int nx = (1 << 11);
    int ny = (1 << 10);

    // int nx = (1 << 10);
    // int ny = (1 << 10);

    // int nx = (1 << 10);
    // int ny = (1 << 9);

    // int nx = 240;
    // int ny = 140;

    Assert((nx & (SIMD_WIDTH - 1)) == 0);
    Assert((ny & (SIMD_WIDTH - 1)) == 0);

    int ns = 1;

    int chunkdim = (1 << 4);
    int chunkx = (int)ceil((float)nx/(float)chunkdim);
    int chunky = (int)ceil((float)ny/(float)chunkdim);

// #if SIMD_PACKET_1X4
//     g_screenbuffer.w = nx + (nx & (SIMD_WIDTH - 1));
//     g_screenbuffer.h = ny;//+ (ny & (SIMD_WIDTH - 1));

#if SIMD_PACKET_2X2
    int w = nx + (nx & 1);
    int h = ny + (ny & 1);

    for (int i=0 ; i<BUFFER_COUNT ; i++)
    {
        g_pixelbuffers[i].w = w;
        g_pixelbuffers[i].h = h;

        // kinda silly, better to alloc dim sized chunks for cache coherency.
        // better still, write out 32bpp values directly.
        int bufsize = w*h*sizeof (float)*ELEMENTS_PER_PIXEL;

        uint64_t g_screenmempoolsize = bufsize*4 + CACHELINE_SIZE*4;
        g_screenmempool.base = (char *) malloc (g_screenmempoolsize);
        g_screenmempool.at = g_screenmempool.base;
        g_screenmempool.remaining = g_screenmempoolsize;

        g_pixelbuffers[i].data = (float *) PoolAlloc (&g_screenmempool, bufsize, CACHELINE_SIZE);
        g_pixelbuffers[i].shouldprint = false;
        g_pixelbuffers[i].postfix = "";
    }

    g_pixelbuffers[COLOR_INDEX].shouldprint = false;
    g_pixelbuffers[OUTPUT_INDEX].shouldprint = true;
    g_pixelbuffers[ALBEDO_INDEX].shouldprint = false;
    g_pixelbuffers[NORMAL_INDEX].shouldprint = false;

    g_pixelbuffers[COLOR_INDEX].postfix = "";
    g_pixelbuffers[OUTPUT_INDEX].postfix = "denoised";
    g_pixelbuffers[ALBEDO_INDEX].postfix = "albedo";
    g_pixelbuffers[NORMAL_INDEX].postfix = "normal";
#endif

      //////////////////////////////////////////////////////////////////////////
     //// Setup Job Queue /////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    // NOTE: (Kapsy) Setup the render chunk data mem.

    // TODO: (Kapsy) Actually work this out.
    g_renderchunkcount = (1 << 10);
    g_renderchunks = (renderchunkdata_t *) PoolAlloc (&g_screenmempool, sizeof (*g_renderchunks)*g_renderchunkcount, CACHELINE_SIZE);

    unsigned int initialcount = 0;
    g_jobqueue.semaphorehandle = dispatch_semaphore_create(initialcount);

    int threadcount = 10;
    for (int i=0 ; i < threadcount ; i++)
    {
        pthread_t pthread = 0;

        int result = pthread_create (&pthread, 0, ThreadProc, 0);
        Assert (result == 0);

        // IOProcThreadPolicySet = 1;
        thread_affinity_policy_data_t policy = {};
        policy.affinity_tag = (int)(i);
        int res = thread_policy_set (pthread_mach_thread_np (pthread), THREAD_AFFINITY_POLICY, (thread_policy_t) &policy, THREAD_AFFINITY_POLICY_COUNT);
    }

    thread_affinity_policy_data_t policy = {};
    policy.affinity_tag = threadcount;
    int res = thread_policy_set (pthread_mach_thread_np (pthread_self()), THREAD_AFFINITY_POLICY, (thread_policy_t) &policy, THREAD_AFFINITY_POLICY_COUNT);

      //////////////////////////////////////////////////////////////////////////
     //// Render Frames ///////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

#define rdtsc __builtin_readcyclecounter

    mach_timebase_info_data_t timebaseinfo;
    mach_timebase_info(&timebaseinfo);
    uint64_t startmachtime = mach_absolute_time();

    long long StartCount = rdtsc();

    // NOTE: (Kapsy) Setup the camera.
    camera cam;

    // v3 lookfromres = lookfrom;
    // NOTE: (Kapsy) Adjust distance based on angle.
    // lookfromres = lookfromres*((-cos(ellipsephase)+1.f)*3.07f + 1.3f);

    m44 lookfromT = {
        1.00f, 0.00f, 0.00f, 0.00f,
        0.00f, 1.00f, 0.00f, 1.9f,
        0.00f, 0.00f, 1.00f, 0.00f,
        0.00f, 0.00f, 0.00f, 1.00f,
    };
    lookfrom = lookfrom*lookfromT;

    v3 vup = V3 (-0.18f,1,0);
    //v3 vup = V3 (0,1,0);
    float vfov = 60.f;
    float aspect = (float)nx/(float)ny;

    float aperture = 0.0733f;

    float focusdist = Length (lookfrom - lookat);

    cam.lensrad = aperture/2.f;

    float theta = vfov*M_PI/180.f;
    float halfheight = tan(theta/2.f);
    float halfwidth = aspect*halfheight;

    cam.origin = lookfrom;
    cam.w = Unit (lookfrom - lookat);
    cam.u = Unit (Cross (vup, cam.w));
    cam.v = Cross (cam.w, cam.u);

    cam.lowerleft = cam.origin - halfwidth*focusdist*cam.u - halfheight*focusdist*cam.v - focusdist*cam.w;
    cam.horiz = 2.f*halfwidth*focusdist*cam.u;
    cam.vert = 2.f*halfheight*focusdist*cam.v;

    // NOTE: (Kapsy) Set the initial focus distance.
    focusdist = GetAutofocusDistance (&cam, object, g_fastbsp, nx, ny);
    // move these to cam
    float targetfocusdist = focusdist;
    //float focusupdaterate = 0.f;


// #define RENDER_FRAMES ((f == 0) || (f == 80) || (f == 98) || (f == 160) || (f == 260) || (f == 360) || (f == 490))
// #define RENDER_FRAMES ((f == 140) || (f == 360))
// #define RENDER_FRAMES ((f >= 0) && (f < 10))
// #define RENDER_FRAMES ((f == 91))
// #define RENDER_FRAMES ((f == 140))
// #define RENDER_FRAMES ((f == 20))
// #define RENDER_FRAMES ((f == 43))
// #define RENDER_FRAMES (((f%40) == 0))
// #define RENDER_FRAMES ((f == 482))
// #define RENDER_FRAMES ((f == 135))
// #define RENDER_FRAMES ((f == 0))
#define RENDER_FRAMES ((1))

    g_stratanglecount = ny*nx*g_rppcount;
    g_stratangles = (float *)malloc (sizeof (float)*g_stratanglecount);

    for(int y=0 ; y<ny ; y++)
    {
        for(int x=0 ; x<nx ; x++)
        {
            for(int i=0 ; i<g_rppcount ; i++)
            {
                int a = y*nx*g_rppcount + x*g_rppcount + i;

                g_stratangles[a] = ((float)i/(float)g_rppcount)*2.f*M_PI;
            }
        }
    }

    for(int y=0 ; y<ny ; y++)
    {
        for(int x=0 ; x<nx ; x++)
        {
            for(int i=0 ; i<(g_rppcount - 1); i++)
            {

                int next = i + 1;
                int swapindex = next + (int)(drand48 ()*(float)(g_rppcount - next));

                int a = y*nx*g_rppcount + x*g_rppcount + i;
                int b = y*nx*g_rppcount + x*g_rppcount + swapindex;

                float temp = g_stratangles[a];
                g_stratangles[a] = g_stratangles[b];
                g_stratangles[b] = temp;
            }
        }
    }

    for (int f=0 ; f<framecount ; f++)
    {

        if (RENDER_FRAMES)
        {
            printf("%03d ", f + 1);
        }


        float tdelta = g_rppinv*shuttertime;

        for (int rppy=0 ; rppy<g_rppy ; rppy++)
        {
            for (int rppx=0 ; rppx<g_rppx ; rppx++)
            {
                SimulateScene (tdelta, cam, object, g_fastbsp, nx, ny, lookfrom, lookat, focusdist, targetfocusdist, vup, halfwidth, halfheight, rotspeed);

                  //////////////////////////////////////////////////////////////
                 //// Render Motion Distributed Rays //////////////////////////
                //////////////////////////////////////////////////////////////

                if (RENDER_FRAMES)
                {
                    float stratposx = ((float)rppx)*g_stratdimx;
                    // Okay thinking we might have to invert these because we start from the bottom of the pixel and go to the top.
                    float stratposy = ((float)rppy)*g_stratdimy;

                    // float lensstratangle = ((float)(rppy*g_rppx + rppx)/(float)g_rppcount)*2.f*M_PI;
                    // Might be better to shuffle the table every frame?
                    //
                    //// int stratindex = (int)(drand48 ()*(float)(stratcount - 1));
                    //// float lensstratangle = stratangles[stratindex];
                    //// stratangles[stratindex] = stratangles[stratcount - 1];
                    //// stratcount--;

                    //// int stratstartindex = f;
                    int rppstart = (rppy*g_rppx + rppx);
                    //// float lensstratangle = g_stratangles[stratindex];

                    // trying square chunks, although might be better to do things by line.
                    // need to try both.
                    renderchunkdata_t *chunkat = g_renderchunks;

                    // int p = 0;
                    // int q = 0;

                    int remainingh = g_pixelbuffers[COLOR_INDEX].h;
                    for (int p=0 ; p<chunky ; p++)
                    {
                        int remainingw = g_pixelbuffers[COLOR_INDEX].w;
                        for (int q=0 ; q<chunkx ; q++)
                        {
                            // Allocate these before render and reuse.
                            renderchunkdata_t *chunk = chunkat++;

                            // Need to assert remainings are % SIMD_WIDTH?
                            // Should always be though, as the buffer will be.
                            chunk->dimx = min (remainingw, chunkdim);
                            chunk->dimy = min (remainingh, chunkdim);
                            chunk->starty = p*chunkdim;
                            chunk->startx = q*chunkdim;
                            chunk->object = object;
                            // Do we need this if global anyway? Could save on the pointer!
                            chunk->buffers = g_pixelbuffers;

                            chunk->nx = nx;
                            chunk->ny = ny;
                            chunk->cam = &cam;
                            chunk->clearbuffer = ((rppy == 0) && (rppx == 0));

                            chunk->stratposx = stratposx;
                            chunk->stratposy = stratposy;
                            chunk->rppstart = rppstart;

                            chunk->randomseed = _mm_set_epi32 (1234*g_seedfactor, 23316*g_seedfactor, 235402*g_seedfactor, 9443*g_seedfactor);
                            g_seedfactor++;

                            AddJobToQueue (&g_jobqueue, JobTypeRenderChunk, chunk);

                            remainingw -= chunkdim;
                        }

                        remainingh -= chunkdim;
                    }

                    CompleteAllJobs ();

                    printf (".");
                    fflush (0);
                }
            }
        }

          //////////////////////////////////////////////////////////////////////
         //// Denoising ///////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        unsigned int w = g_pixelbuffers[COLOR_INDEX].w;
        unsigned int h = g_pixelbuffers[COLOR_INDEX].h;

        if (RENDER_FRAMES)
        {
            // Create an Open Image Denoise device
            OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
            oidnCommitDevice(device);
            // Create a denoising filter
            OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
            oidnSetSharedFilterImage(filter, "color",  g_pixelbuffers[COLOR_INDEX].data, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0);
            oidnSetSharedFilterImage(filter, "albedo", g_pixelbuffers[ALBEDO_INDEX].data, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0); // optional
            oidnSetSharedFilterImage(filter, "normal", g_pixelbuffers[NORMAL_INDEX].data, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0); // optional
            oidnSetSharedFilterImage(filter, "output", g_pixelbuffers[OUTPUT_INDEX].data, OIDN_FORMAT_FLOAT3, w, h, 0, 0, 0);
            oidnSetFilter1b(filter, "hdr", false); // image is HDR
            oidnCommitFilter(filter);

            // Filter the image
            oidnExecuteFilter(filter);

            // Check for errors
            const char* errorMessage;
            if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
            {
                printf ("Error: %s\n", errorMessage);
                return (EXIT_FAILURE);
            }

            // Cleanup
            oidnReleaseFilter(filter);
            oidnReleaseDevice(device);

            printf ("d");
            fflush (0);
        }

          //////////////////////////////////////////////////////////////////////
         //// Tone Mapping ////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        // NOTE: (Kapsy) Could do other post stuff here too...
        if (RENDER_FRAMES)
        {
            {
                // NOTE: (Kapsy) Trying square chunks, although might be better to do things by line.
                // need to try both.
                renderchunkdata_t *chunkat = g_renderchunks;

                // int p = 0;
                // int q = 0;

                int remainingh = g_pixelbuffers[COLOR_INDEX].h;
                for (int p=0 ; p<chunky ; p++)
                {
                    int remainingw = g_pixelbuffers[COLOR_INDEX].w;
                    for (int q=0 ; q<chunkx ; q++)
                    {
                        // Allocate these before render and reuse.
                        renderchunkdata_t *chunk = chunkat++;

                        // Need to assert remainings are % SIMD_WIDTH?
                        // Should always be though, as the buffer will be.
                        chunk->dimx = min (remainingw, chunkdim);
                        chunk->dimy = min (remainingh, chunkdim);
                        chunk->starty = p*chunkdim;
                        chunk->startx = q*chunkdim;
                        chunk->object = object;
                        chunk->buffers = g_pixelbuffers;

                        chunk->nx = nx;
                        chunk->ny = ny;
                        chunk->cam = &cam;
                        chunk->clearbuffer = 0;

                        AddJobToQueue (&g_jobqueue, JobTypeToneMap, chunk);

                        remainingw -= chunkdim;
                    }

                    remainingh -= chunkdim;
                }

                CompleteAllJobs ();
            }

            printf("OK\n");
            fflush (0);


              //////////////////////////////////////////////////////////////////
             //// Write Image /////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////

            for (int i=0 ; i<BUFFER_COUNT ; i++)
            {
                pixelbuffer_t *pixelbuffer = g_pixelbuffers + i;

                if (pixelbuffer->shouldprint)
                {
                    char filename[(1 << 6)];
                    sprintf(filename, "temp/out_%03d%s.ppm", (f+1), pixelbuffer->postfix);
                    FILE *frame = fopen (filename, "w");

                    if (frame)
                    {
                        float *bufferat = pixelbuffer->data;
                        fprintf (frame, "P3\n%d %d\n255\n", nx, ny);
                        for (int j=(ny-1) ; j >= 0 ; j--)
                        {
                            for (int i=0 ; i < nx ; i++)
                            {
                                v3 col;
                                col.r = *bufferat++;
                                col.g = *bufferat++;
                                col.b = *bufferat++;

                                int ir = (int)(255.99*Clamp01 (col.r));
                                int ig = (int)(255.99*Clamp01 (col.g));
                                int ib = (int)(255.99*Clamp01 (col.b));

                                fprintf(frame, "%d %d %d\n",  ir, ig, ib);
                            }
                        }

                        fclose (frame);
                    }
                }
            }
        }

          //////////////////////////////////////////////////////////////////////
         //// Simulate Rest of Frame //////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        tdelta = frametime - shuttertime;
        SimulateScene (tdelta, cam, object, g_fastbsp, nx, ny, lookfrom, lookat, focusdist, targetfocusdist, vup, halfwidth, halfheight, rotspeed);
    }

    long long EndCount = rdtsc ();
    float megacycles = (float)((EndCount - StartCount) / (1000.f * 1000.f));

    uint64_t endmachtime = mach_absolute_time();
    uint64_t rendertimens =
        (endmachtime - startmachtime)*
        (timebaseinfo.numer/timebaseinfo.denom);
    float rendertimesecs = (float)rendertimens/(float)NSPerS;

    numrays = numprimaryrays + numsecondaryrays;
    uint64_t framecount_int = (uint64_t)framecount;

    printf ("\nScene stats:\n");
    printf ("Megacycles:%f\n", megacycles);
    printf ("Render time total:%f\n", rendertimesecs);
    printf ("Frame count: %llu\n", framecount_int);
    printf ("Triangles: %llu\n", numtris);
    printf ("Rays/s: %f\n", numrays/rendertimesecs);

    printf ("\nPer frame stats:\n");
    printf ("Render time: %fs\n", rendertimesecs/framecount_int);
    printf ("Primary rays: %llu\n", numprimaryrays/framecount_int);
    printf ("Split packet rays: %llu\n", numsplitpacketrays/framecount_int);
    printf ("Split packet ratio: %f\n", (float)numsplitpacketrays/(float)numprimaryrays/(float)framecount_int);

    printf ("Secondary rays: %llu\n", numsecondaryrays/framecount_int);
    printf ("Total Rays: %llu\n", numrays/framecount_int); // needed?

    printf ("Total packets: %llu\n", g_numpackets);
    printf ("Split packets: %llu\n", g_numsplitpackets);

    //// printf("Ray triangle tests: %llu\n", numraytritests);
    //// printf("Ray triangle intersections: %llu\n", numraytriintersections);
    //// printf("Triangle hit ratio: %f\n", (float)((double)numraytriintersections/(double)numraytritests));

    rendertimesecs = 0.f;
    numtris = 0;
    numprimaryrays = 0;
    numsecondaryrays = 0;
    numrays = 0;
    numraytritests = 0;
    numraytriintersections = 0;
    numboundingboxtests = 0;
    numboundingboxintersections = 0;

    return(EXIT_SUCCESS);
}
