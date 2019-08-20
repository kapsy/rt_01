
#define Rad(phi) (M_PI*(phi/180.f))

#define EPSILON 0.0000001f

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

// NOTE: (Kapsy) Convenience macro functions.
#define Inv(a) (_mm_xor_ps (a, _mm_set1_epi32 (0xffffffff)))
#define And3(a, b, c) (_mm_and_ps (_mm_and_ps (a, b), c))
#define OrE(a, b) (a = _mm_or_ps (a, b))


#define MM_ZERO _mm_set1_ps (0.f)
#define MM_HALF _mm_set1_ps (0.5f)
#define MM_ONE _mm_set1_ps (1.f)
#define MM_TWO _mm_set1_ps (2.f)
#define MM_NEGONE _mm_set1_ps (-1.f)
#define MM_ALL _mm_set1_epi32 (0xffffffff)
#define MM_INV(v) (_mm_xor_ps (v, MM_ALL))

inline float
Clamp01 (float a)
{
    float res = a;

    if (res > 1.f)
        res = 1.f;
    else if (res < 0.f)
        res = 0.f;

    return (res);
}

inline float
Clamp (float min, float max, float a)
{
    float res = a;

    if (res > max)
        res = max;
    else if (res < min)
        res = min;

    return (res);
}

inline void
Swap (float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

// NOTE: (Kapsy) THIS IS MIT LICENSE CODE, WILL HAVE TO REMOVE AT SOME POINT.
#define EXP_POLY_DEGREE 3

#define POLY0(x, c0) _mm_set1_ps(c0)
#define POLY1(x, c0, c1) _mm_add_ps(_mm_mul_ps(POLY0(x, c1), x), _mm_set1_ps(c0))
#define POLY2(x, c0, c1, c2) _mm_add_ps(_mm_mul_ps(POLY1(x, c1, c2), x), _mm_set1_ps(c0))
#define POLY3(x, c0, c1, c2, c3) _mm_add_ps(_mm_mul_ps(POLY2(x, c1, c2, c3), x), _mm_set1_ps(c0))
#define POLY4(x, c0, c1, c2, c3, c4) _mm_add_ps(_mm_mul_ps(POLY3(x, c1, c2, c3, c4), x), _mm_set1_ps(c0))
#define POLY5(x, c0, c1, c2, c3, c4, c5) _mm_add_ps(_mm_mul_ps(POLY4(x, c1, c2, c3, c4, c5), x), _mm_set1_ps(c0))

m128 exp2f4(__m128 x)
{
   __m128i ipart;
   __m128 fpart, expipart, expfpart;

   x = _mm_min_ps(x, _mm_set1_ps( 129.00000f));
   x = _mm_max_ps(x, _mm_set1_ps(-126.99999f));

   /* ipart = int(x - 0.5) */
   ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _mm_set1_ps(0.5f)));

   /* fpart = x - ipart */
   fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));

   /* expipart = (float) (1 << ipart) */
   expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _mm_set1_epi32(127)), 23));

   /* minimax polynomial fit of 2**x, in range [-0.5, 0.5[ */
#if EXP_POLY_DEGREE == 5
   expfpart = POLY5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
#elif EXP_POLY_DEGREE == 4
   expfpart = POLY4(fpart, 1.0000026f, 6.9300383e-1f, 2.4144275e-1f, 5.2011464e-2f, 1.3534167e-2f);
#elif EXP_POLY_DEGREE == 3
   expfpart = POLY3(fpart, 9.9992520e-1f, 6.9583356e-1f, 2.2606716e-1f, 7.8024521e-2f);
#elif EXP_POLY_DEGREE == 2
   expfpart = POLY2(fpart, 1.0017247f, 6.5763628e-1f, 3.3718944e-1f);
#else
#error
#endif

   return _mm_mul_ps(expipart, expfpart);
}


#define LOG_POLY_DEGREE 5

__m128 log2f4(__m128 x)
{
   __m128i exp = _mm_set1_epi32(0x7F800000);
   __m128i mant = _mm_set1_epi32(0x007FFFFF);

   __m128 one = _mm_set1_ps( 1.0f);

   __m128i i = _mm_castps_si128(x);

   __m128 e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(_mm_and_si128(i, exp), 23), _mm_set1_epi32(127)));

   __m128 m = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(i, mant)), one);

   __m128 p;

   /* Minimax polynomial fit of log2(x)/(x - 1), for x in range [1, 2[ */
#if LOG_POLY_DEGREE == 6
   p = POLY5( m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f,  3.1821337e-1f, -3.4436006e-2f);
#elif LOG_POLY_DEGREE == 5
   p = POLY4(m, 2.8882704548164776201f, -2.52074962577807006663f, 1.48116647521213171641f, -0.465725644288844778798f, 0.0596515482674574969533f);
#elif LOG_POLY_DEGREE == 4
   p = POLY3(m, 2.61761038894603480148f, -1.75647175389045657003f, 0.688243882994381274313f, -0.107254423828329604454f);
#elif LOG_POLY_DEGREE == 3
   p = POLY2(m, 2.28330284476918490682f, -1.04913055217340124191f, 0.204446009836232697516f);
#else
#error
#endif

   /* This effectively increases the polynomial degree by one, but ensures that log2(1) == 0*/
   p = _mm_mul_ps(p, _mm_sub_ps(m, one));

   return _mm_add_ps(p, e);
}

static inline __m128
powf4(__m128 x, __m128 y)
{
   return exp2f4(_mm_mul_ps(log2f4(x), y));
}

  //////////////////////////////////////////////////////////////////////////////
 //// Vector Math /////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

inline m128
Min4 (m128 a, m128 b)
{
    m128 mask = (a < b);
    m128 maskinv = _mm_xor_ps(mask, _mm_set1_epi32(0xffffffff));
    m128 res = _mm_and_ps(a, mask) + _mm_and_ps(b, maskinv);
    return (res);
}

inline m128
Max4 (m128 a, m128 b)
{
    m128 mask = (a > b);
    m128 maskinv = _mm_xor_ps(mask, _mm_set1_epi32(0xffffffff));
    // TODO: (Kapsy) Could use an or instead of a + here, don't think that it really matters that much.
    m128 res = _mm_and_ps(a, mask) + _mm_and_ps(b, maskinv);
    return (res);
}

inline m128
Clamp014 (m128 a)
{
    m128 res = a;

    m128 zero = _mm_set1_ps(0.f);
    m128 one = _mm_set1_ps(1.f);

    // TODO: (Kapsy) This *should* work but I need to check...
    res = Min4 (res, one);
    res = Max4 (res, zero);

    return (res);
}

inline m128
Clamp4 (m128 a, m128 min, m128 max)
{
    m128 res = a;

    // TODO: (Kapsy) This *should* work but I need to check...
    res = Min4 (res, max);
    res = Max4 (res, min);

    return (res);
}



inline bool
SignsMatch (m128 a)
{
    unsigned int A = ((*(((unsigned int *)&a) + 0)) & 0x80000000);
    unsigned int B = ((*(((unsigned int *)&a) + 1)) & 0x80000000);
    unsigned int C = ((*(((unsigned int *)&a) + 2)) & 0x80000000);
    unsigned int D = ((*(((unsigned int *)&a) + 3)) & 0x80000000);

    bool res = (A == B) && (A == C) && (A == D);

    return (res);
}

inline bool
HaveBitsSet (m128 a)
{
    // TODO: (Kapsy) This is dumb, there are better ways to do this.
    unsigned int merged = (
            (*(((unsigned int *)&a) + 0)) |
            (*(((unsigned int *)&a) + 1)) |
            (*(((unsigned int *)&a) + 2)) |
            (*(((unsigned int *)&a) + 3)));

    bool res = (merged != 0);

    return (res);
}


// TODO: (Kapsy) Should find a faster way to do this! Slow...
inline bool
AllBitsSet4 (m128 a)
{
    unsigned int merged = (
            (*(((unsigned int *)&a) + 0)) &
            (*(((unsigned int *)&a) + 1)) &
            (*(((unsigned int *)&a) + 2)) &
            (*(((unsigned int *)&a) + 3)));

    bool res = (merged == 0xffffffff);
    return (res);
}

inline bool
NoBitsSet4 (m128 a)
{
    unsigned int merged = (
            (*(((unsigned int *)&a) + 0)) |
            (*(((unsigned int *)&a) + 1)) |
            (*(((unsigned int *)&a) + 2)) |
            (*(((unsigned int *)&a) + 3)));

    bool res = (merged == 0);
    return (res);
}


inline int
BitsSetCount (m128 a)
{
    int res = 0;

    res += (*(((unsigned int *)&a) + 0) == 0xffffffff);
    res += (*(((unsigned int *)&a) + 1) == 0xffffffff);
    res += (*(((unsigned int *)&a) + 2) == 0xffffffff);
    res += (*(((unsigned int *)&a) + 3) == 0xffffffff);

    return (res);
}


inline int
BitsSetIndex (m128 a)
{
    int res = 0;

    // NOTE: (Kapsy) No checks for now! Assumes only one element set.
    if (*(((unsigned int *)&a) + 0) == 0xffffffff)
        res = 0;
    if (*(((unsigned int *)&a) + 1) == 0xffffffff)
        res = 1;
    if (*(((unsigned int *)&a) + 2) == 0xffffffff)
        res = 2;
    if (*(((unsigned int *)&a) + 3) == 0xffffffff)
        res = 3;

    return (res);
}


union v3
{
    struct
    {
        float x, y, z;
    };

    struct
    {
        float r, g, b;
    };

    struct
    {
        float u, v, w;
    };

    float e[3];
};

inline v3
V3(float x, float y, float z)
{
    v3 res = { x, y, z };
    return (res);
}

inline v3
V3(float a)
{
    v3 res = { a, a, a };
    return (res);
}

inline v3
operator*(const float a, const v3 &b)
{
    v3 res = { a*b.e[0], a*b.e[1], a*b.e[2] };

    return (res);
}

inline v3
operator*(const v3 &a, const float &b)
{
    v3 res = { a.e[0]*b, a.e[1]*b, a.e[2]*b };

    return (res);
}

inline v3
operator-(const v3 &a)
{
    v3 res = { -a.e[0], -a.e[1], -a.e[2] };

    return (res);
}

inline v3
operator+(const v3 &a, const v3 &b)
{
    v3 res = { a.e[0] + b.e[0], a.e[1] + b.e[1], a.e[2] + b.e[2] };

    return (res);
}

inline v3
operator-(const v3 &a, const v3 &b)
{
    v3 res = { a.e[0] - b.e[0], a.e[1] - b.e[1], a.e[2] - b.e[2] };

    return (res);
}

inline v3
operator*(const v3 &a, const v3 &b)
{
    v3 res = { a.e[0] * b.e[0], a.e[1] * b.e[1], a.e[2] * b.e[2] };

    return (res);
}

inline v3
operator/(const v3 &a, const float &b)
{
    v3 res = { a.e[0]/b, a.e[1]/b, a.e[2]/b };
    return (res);
}

inline v3
operator/(const float &b, const v3 &a)
{
    v3 res = { b/a.e[0], b/a.e[1], b/a.e[2] };
    return (res);
}

inline v3 &
operator/=(v3 &a, float b)
{
    a = a/b;
    return (a);
}

inline v3 &
operator+=(v3 &a, v3 b)
{
    a = a + b;
    return (a);
}

//// inline m128
//// operator&(const m128 &a, const m128 &b)
//// {
////     m128 res = _mm_and_ps (a, b);
////     return (res);
//// }

inline float
Dot (const v3 &a, const v3 &b)
{
    float res = (a.e[0]*b.e[0] + a.e[1]*b.e[1] + a.e[2]*b.e[2]);
    return (res);
}

// NOTE: (Kapsy) Wikipedia definition.
// a2b3 - a3b2
// a3b1 - a1b3
// a1b2 - a2b1
inline v3
Cross (const v3 &a, const v3 &b)
{
    v3 res = V3 (
            a.e[1]*b.e[2] - a.e[2]*b.e[1],
            a.e[2]*b.e[0] - a.e[0]*b.e[2],
            a.e[0]*b.e[1] - a.e[1]*b.e[0]);

    return (res);
}


inline v3
Unit (const v3 &a)
{
    v3 res = a / sqrt (Dot (a, a));
    return (res);
}

inline float
Length (const v3 &a)
{
    float res = sqrt (Dot (a, a));
    return (res);
}

inline float
SquaredLen (v3 &a)
{
    return (Dot (a, a));
}

  //////////////////////////////////////////////////////////////////////////////
 //// SIMD Vector Math /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

union v34
{
    struct
    {
        m128 x, y, z;
    };

    struct
    {
        m128 r, g, b;
    };

    m128 e[4];
};


inline v34
V34(m128 x, m128 y, m128 z)
{
    v34 res = { x, y, z };
    return (res);
}

inline v34
V34(m128 a)
{
    v34 res = {
        a,
        a,
        a,
    };

    return (res);
}

inline v34
V34 (const v3 &a)
{
    v34 res = {
        _mm_set1_ps(a.x),
        _mm_set1_ps(a.y),
        _mm_set1_ps(a.z),
    };

    return (res);
}

inline v34
operator/(m128 a, const v34 &b)
{
    v34 res = {
        a/b.x,
        a/b.y,
        a/b.z,
    };

    return (res);
}

inline
v34 operator-(const v34 &a)
{
    v34 res;

    m128 negone = _mm_set1_ps (-1.f);

    res.x = a.x*negone;
    res.y = a.y*negone;
    res.z = a.z*negone;


    return (res);
}

// TODO: (Kapsy) As a rule, should only allow m128s as inputs to V34 functions.
// The conversion has to be done internally anyway.
inline v34
operator*(const v34 &a, m128 b)
{
    v34 res = {
        a.x*b,
        a.y*b,
        a.z*b,
    };

    return (res);
}

inline v34
operator*(m128 a, const v34 &b)
{
    return (b*a);
}

inline v34
operator*(const v34 &a, const v34 &b)
{
    v34 res = {
        a.x*b.x,
        a.y*b.y,
        a.z*b.z,
    };

    return (res);
}

inline v34
operator+(const v34 &a, const v34 &b)
{
    v34 res = {
        a.e[0] + b.e[0],
        a.e[1] + b.e[1],
        a.e[2] + b.e[2] };

    return (res);
}

inline v34 &
operator+=(v34 &a, const v34 &b)
{
    a = a + b;

    return (a);
}

inline v34
operator-(const v34 &a, const v34 &b)
{
    v34 res = {
        a.e[0] - b.e[0],
        a.e[1] - b.e[1],
        a.e[2] - b.e[2] };

    return (res);
}


inline v34
operator/(const v34 &a, m128 b)
{
    v34 res = {
        a.x/b,
        a.y/b,
        a.z/b
    };

    return (res);
}

inline v34
operator&(const v34 &a, m128 b)
{
    v34 res = {
        _mm_and_ps (a.x, b),
        _mm_and_ps (a.y, b),
        _mm_and_ps (a.z, b)
    };

    return (res);
}

inline v34
operator&(m128 a, const v34 &b)
{
    return (b & a);
}


inline m128
Dot (const v34 &a, const v34 &b)
{
    m128 res = (a.x*b.x + a.y*b.y + a.z*b.z);

    return (res);
}

inline v34
Unit (const v34 &a)
{
    v34 res = a / _mm_sqrt_ps (Dot (a, a));
    return (res);
}

inline v34
Cross (const v34 &a, const v34 &b)
{
    v34 res = V34 (
            a.e[1]*b.e[2] - a.e[2]*b.e[1],
            a.e[2]*b.e[0] - a.e[0]*b.e[2],
            a.e[0]*b.e[1] - a.e[1]*b.e[0]);

    return (res);
}


inline m128
Length (const v34 &a)
{
    m128 res = _mm_sqrt_ps (Dot (a, a));
    return (res);
}

inline v34
ClampUnit (const v34 &a)
{
    m128 len = Length (a);
    m128 lenover = len > MM_ONE;

    v34 res = ((a/len) & lenover) + (a & MM_INV (lenover));

    return (res);
}

inline m128
SquaredLen4 (v34 &a)
{
    return (Dot (a, a));
}

inline v34
Clamp01 (const v34 &a)
{
    v34 res = V34 (
            Clamp014 (a.x),
            Clamp014 (a.y),
            Clamp014 (a.z)
            );
    return (res);
}

inline v34
Clamp (const v34 &a, m128 min, m128 max)
{
    v34 res = V34 (
            Clamp4 (a.x, min, max),
            Clamp4 (a.y, min, max),
            Clamp4 (a.z, min, max)
            );
    return (res);
}

inline v34
ClampToNormal (v34 r, v34 N)
{
    v34 rclamp = Unit (Cross (N, Cross (N, r)));
    m128 anglemask = (Dot (N, r) >= MM_ZERO);
    v34 res = (r & anglemask) + (rclamp & MM_INV (anglemask));

    return (res);
}


  //////////////////////////////////////////////////////////////////////////////
 //// Matrix Math /////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

union m44
{
    struct
    {
        float a, b, c, d;
        float e, f, g, h;
        float i, j, k, l;
        float m, n, o, p;
    };

    float E[4][4];
};

inline m44
M44Scale (float xyz)
{
    m44 res = {
        xyz, 0.f, 0.f, 0.f,
        0.f, xyz, 0.f, 0.f,
        0.f, 0.f, xyz, 0.f,
        0.f, 0.f, 0.f, 1.f,
    };
    return (res);
}

inline m44
M44Trans (float x, float y, float z)
{
    m44 res = {
        1.f, 0.f, 0.f,   x,
        0.f, 1.f, 0.f,   y,
        0.f, 0.f, 1.f,   z,
        0.f, 0.f, 0.f, 1.f,
    };
    return (res);
}

inline m44
M44RotX (float phi)
{
    float cosp = cos(phi);
    float sinp = sin(phi);

    m44 res = {
        1.f,   0.f,   0.f, 0.f,
        0.f,  cosp, -sinp, 0.f,
        0.f,  sinp,  cosp, 0.f,
        0.f,   0.f,   0.f, 1.f,
    };

    return (res);
}

inline m44
M44RotY (float phi)
{
    float cosp = cos(phi);
    float sinp = sin(phi);

    m44 res = {
        cosp, 0.f, sinp, 0.f,
         0.f, 1.f,  0.f, 0.f,
       -sinp, 0.f, cosp, 0.f,
         0.f, 0.f,  0.f, 1.f,
    };

    return (res);
}

inline m44
M44RotZ (float phi)
{
    float cosp = cos(phi);
    float sinp = sin(phi);

    m44 res = {
        cosp,-sinp, 0.f, 0.f,
        sinp, cosp, 0.f, 0.f,
         0.f,  0.f, 1.f, 0.f,
         0.f,  0.f, 0.f, 1.f,
    };

    return (res);
}

inline void
DebugPrint(m44 *A)
{
    int n = 4;
    int m = 4;

    for (int i=0 ; i<n ; i++)
    {
        for (int j=0 ; j<m ; j++)
        {
            printf("%.02f, ", A->E[i][j]);
        }

        printf("\n");
    }

    printf("\n");
}

// NOTE: (Kapsy) vector matrix multiplication:
//         x
//         y
//         z
// a b c d ax + by + cz + d
// e f g h ex + fy + gz + h
// i j k l ix + jy + kz + l
inline v3
operator*(const v3 &A, const m44 &B)
{
    v3 res = {};

    res.x = B.a*A.x + B.b*A.y + B.c*A.z + B.d;
    res.y = B.e*A.x + B.f*A.y + B.g*A.z + B.h;
    res.z = B.i*A.x + B.j*A.y + B.k*A.z + B.l;

    return (res);
}

inline v3
operator*(const m44 &B, const v3 &A)
{
    return (A*B);
}

inline m44
operator*(const m44 &A, const m44 &B)
{
    m44 res = {};

    int n = 4;
    int m = 4;

    for (int i=0 ; i<n ; i++)
    {
        for (int j=0 ; j<m ; j++)
        {
            float e = 0.f;

            e += A.E[i][0]*B.E[0][j];
            e += A.E[i][1]*B.E[1][j];
            e += A.E[i][2]*B.E[2][j];
            e += A.E[i][3]*B.E[3][j];

            res.E[i][j] = e;
        }
    }

    return (res);
}

  //////////////////////////////////////////////////////////////////////////////
 //// Wide Matrix /////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

union m444
{
    struct
    {
        m128 a, b, c, d;
        m128 e, f, g, h;
        m128 i, j, k, l;
        m128 m, n, o, p;
    };

    m128 E[4][4];
};

inline v34
operator*(const v34 &A, const m444 &B)
{
    v34 res = {};

    res.x = B.a*A.x + B.b*A.y + B.c*A.z + B.d;
    res.y = B.e*A.x + B.f*A.y + B.g*A.z + B.h;
    res.z = B.i*A.x + B.j*A.y + B.k*A.z + B.l;

    return (res);
}

inline v34
operator*(const m444 &B, const v34 &A)
{
    return (A*B);
}

  //////////////////////////////////////////////////////////////////////////////
 //// Rect 3 //////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

union rect3
{
    struct
    {
        v3 min, max;
    };

    v3 e[2];
};

static rect3 boundingbox;

inline bool
PointInRect (rect3 b, v3 a)
{
    bool res = (((b.min.x <= a.x) && (a.x <= b.max.x)) &&
                ((b.min.y <= a.y) && (a.y <= b.max.y)) &&
                ((b.min.z <= a.z) && (a.z <= b.max.z)));

    return (res);
}

///

#define NSPerUS (1000.0)
#define NSPerMS (1000.0 * NSPerUS)

// TODO: (Kapsy) Just use these guys!
#define SToUS(Value) ((Value) * USPerS)
#define SToNS(Value) (Value * NSPerS)

#define MSToS(Value)
#define MSToUS(Value) ((Value) * USPerMS)
#define MSToNS(Value) (Value * NSPerMS)

#define USToS(Value)
#define USToNS(Value)

#define NSToS(Value) ((Value) / NSPerS)
#define NSToMS(Value) ((Value) / NSPerMS)
#define NSToUS(Value) ((Value) / NSPerUS)

#define Kilobytes(Value) ((Value)*1024LL)
#define Megabytes(Value) (Kilobytes(Value)*1024LL)
#define Gigabytes(Value) (Megabytes(Value)*1024LL)
#define Terabytes(Value) (Gigabytes(Value)*1024LL)
#define NSPerS  (1000.0 * NSPerMS)

#define USPerMS (1000.0)
#define USPerS  (1000.0 * USPerMS)
