
  //////////////////////////////////////////////////////////////////////////////
 //// Tri Hit Accelaration Structs ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// NOTE: (Kapsy) For triangle intersection acceleration.
struct triacc
{
    // plane:
    float n_u; // == normal.u/normal.k
    float n_v; // == normal.v/normal.k
    float n_d; // constant of plane equation.
    int k;

    // line equation for line ac
    float b_nu;
    float b_nv;
    float b_d;
    int pad0;

    // line equation for ab
    float c_nu;
    float c_nv;
    float c_d;
    int index;

    // padding to 64 bytes
    int pad2;
    int pad3;
    int pad4;
    int pad5;
};

// NOTE: (Kapsy) This is really dumb, better to use triacc and _mm_set1_ps for the sake of cache.
struct triacc4
{
    // plane:
    m128 n_u; // == normal.u/normal.k
    m128 n_v; // == normal.v/normal.k
    m128 n_d; // constant of plane equation.
    int k; // assumes all ks are the same??? yes, makes sense, we test 4 rays against one tri at a time.
    int padi_0;
    int padi_1;
    int padi_2;

    // line equation for line ac
    m128 b_nu;
    m128 b_nv;
    m128 b_d;
    m128 pad0;

    // line equation for ab
    m128 c_nu;
    m128 c_nv;
    m128 c_d;
    m128 pad1;
};

struct triacclist_t
{
    //triacc *triaccs;
    p32 triaccs;

    int count;
};

  //////////////////////////////////////////////////////////////////////////////
 //// AA BSP Tree Structs /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// NOTE: (Kapsy) From Wald PHD:
// "Note that the exact order and arrangement of the bits has been very
// carefully designed: Each value can be extracted by exactly one “bitwise and”
// operation to mask out the other bits, and does not require any costly shift
// operations for shifting bits to their correct positions."

struct bsp_inner
{
    unsigned int flagoffset;
    // bits 0..1  splitting dimension
    // bits 2..30 offset bits
    // bit  31    leaf flag

    float split; // split co-ord for the nodes dim
};

// NOTE: (Kapsy)
// For leaf nodes, an “item list”, i.e. a list of integer IDs that specify the
// triangles in this leaf; consists of a pointer (or index) to the first item
// in the list, and of the number of items in the list
// so this implies an intermediate list that then contains another pointer.
// let's just do that.
// If we had a node index we could just use that.

struct bsp_leaf
{
    unsigned int flagoffset;
    // bits 2..30 offset to the first child
    // bit  31    leaf flag

    float pad; // not used  could store count here
    //unsigned int tricount;
};

union bsp_node
{
    struct
    {
        unsigned int flagoffset;
        float dontuse;
    };

    bsp_leaf leaf;
    bsp_inner inner;
};

#define BSP_IsLeaf(n) (n->flagoffset & (unsigned int)(1<<31))
#define BSP_Dim(n)    (n->flagoffset & 0x3)
#define BSP_Offset(n) (n->flagoffset & 0x7ffffffc)

#define SetBSPType(n, t)   (n->flagoffset |= ((t & 0x1) << 31))
#define SetBSPDim(n, d)    (n->flagoffset |= (d & 0x3))
#define SetBSPOffset(n, o) (n->flagoffset |= (o & 0x7ffffffc))

struct stack_item
{
    bsp_node *node;
    float tnear;
    float tfar;
};

struct stackitem4_t
{
    bsp_node *node;
    m128 tnear;
    m128 tfar;
};

#define MAX_BSP_STACK_COUNT (1 << 7)
//static stack_item bspstack[MAX_BSP_STACK_COUNT];
//static int bspstackcount;

// static stackitem4_t g_bspstack4[MAX_BSP_STACK_COUNT];
// static int g_bspstack4count;

  //////////////////////////////////////////////////////////////////////////////
 //// Fast BSP Structs ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define MAX_BSP_NODES (1 << 25)
#define BSP_FILENAME "test.bsp"

// TODO: (Kapsy) Should we just use the same struct for a file?
#pragma pack(push, 1)
struct fastbsp_t
{
    unsigned int nodecount;
    p32 nodes;

    unsigned int triacclistcount;
    p32 triacclist;

    unsigned int triacccount;
    p32 triaccsbase;
    // needed here?
    //triacc *triaccat;

    //int *triaccindexes;
    //int *triaccindexat; // needed?

    //char data[1];
};
#pragma pack(pop)

static fastbsp_t *g_fastbsp;

  //////////////////////////////////////////////////////////////////////////////
 //// Tri Hit Accelaration Functions //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// ALIGN(ALIGN_CACHELINE) static const
// Need pragma align...
unsigned int modulo[] = {0,1,2,0,1};

// TODO: (Kapsy) Should store these in the object?
static void
CreateTriacc(object_t *object, int triindex, triacc *accat)
{
    // NOTE: (Kapsy) Init triangle accelaration structures.

    tri_t *tri = object->tris + triindex;
    accat->index = triindex;

    v3 A = object->verts[tri->A];
    v3 B = object->verts[tri->B];
    v3 C = object->verts[tri->C];

    v3 b = C - A;
    v3 c = B - A;

    // Should already have.
    v3 N = Cross (c, b);

    float Nx = abs (N.x);
    float Ny = abs (N.y);
    float Nz = abs (N.z);

    int k =  0;
    if (Nx > Ny)
    {
        if (Nx > Nz) { k = 0; } // X
        else         { k = 2; } // Z
    }
    else
    {
        if (Ny > Nz) { k = 1; } // Y
        else         { k = 2; } // Z
    }

    int u = modulo[k + 1];
    int v = modulo[k + 2];

    // Plane
    float Ninv = 1.0/N.e[k];
    accat->n_u = N.e[u]*Ninv;
    accat->n_v = N.e[v]*Ninv;
    accat->n_d = Dot (N, A)*Ninv;
    accat->k = k;

    float denom = 1.f/(b.e[u]*c.e[v] - b.e[v]*c.e[u]);

    // Line equation for line ac
    accat->b_nu = b.e[u]*denom;
    accat->b_nv = -b.e[v]*denom;
    accat->b_d = (b.e[v]*A.e[u] - b.e[u]*A.e[v])*denom;

    // Line equation for line ab
    accat->c_nu = c.e[v]*denom;
    accat->c_nv = -c.e[u]*denom;
    accat->c_d = (c.e[u]*A.e[v] - c.e[v]*A.e[u])*denom;

#if 0
    // Plane
    float Ninv = 1.0/N.e[k];
    accat->n_u = _mm_set1_ps(N.e[u]*Ninv);
    accat->n_v = _mm_set1_ps(N.e[v]*Ninv);
    accat->n_d = _mm_set1_ps(Dot (N, A)*Ninv);
    accat->k = k;

    float denom = 1.f/(b.e[u]*c.e[v] - b.e[v]*c.e[u]);

    // line equation for line ac
    accat->b_nu = _mm_set1_ps(b.e[u]*denom);
    accat->b_nv = _mm_set1_ps(-b.e[v]*denom);
    accat->b_d = _mm_set1_ps((b.e[v]*A.e[u] - b.e[u]*A.e[v])*denom);

    // line equation for line ab
    accat->c_nu = _mm_set1_ps(c.e[v]*denom);
    accat->c_nv = _mm_set1_ps(-c.e[u]*denom);
    accat->c_d = _mm_set1_ps((c.e[u]*A.e[v] - c.e[v]*A.e[u])*denom);
#endif
}

  //////////////////////////////////////////////////////////////////////////////
 //// Wald Intersection ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Should move this stuff to a new source file.

inline void
WaldIntersection(triacc *acc, ray *r, hitrec *hit)
{
#define ku modulo[acc->k+1]
#define kv modulo[acc->k+2]

    // Start high latency division early on - likely to get reordered?
    const float nd = 1.f/(r->dir.e[acc->k] + acc->n_u*r->dir.e[ku] + acc->n_v*r->dir.e[kv]);
    const float f = (acc->n_d - r->orig.e[acc->k] - acc->n_u*r->orig.e[ku] - acc->n_v*r->orig.e[kv])*nd;

    // Check for valid distance
    if (!(hit->dist > f && f > EPSILON))
        return;

    // Compute hit point positions on uv plane
    const float hu = (r->orig.e[ku] + f*r->dir.e[ku]);
    const float hv = (r->orig.e[kv] + f*r->dir.e[kv]);

    // Check first barycentric coordinate
    const float lambda = (hv*acc->b_nu + hu*acc->b_nv + acc->b_d);
    if (lambda < 0.f)
        return;

    // Check second barycentric coordinate
    const float mue = (hu*acc->c_nu + hv*acc->c_nv + acc->c_d);
    if (mue < 0.f)
        return;

    // Check third barycentric coordinate
    if ((lambda + mue) > 1.f)
        return;

    numraytriintersections++;
    // Have a valid hitpoint here, store it.
    hit->dist = f;
    hit->primref = acc->index;
    hit->primtype = PrimTypeTri;
    hit->u = lambda;
    hit->v = mue;
}

// hits not working. The smaller the tris, the worse it seems to be
// could try using WaldIntersection1 to see if it still occurs.
inline void
WaldIntersection4(triacc *acc, ray4 *r, hitrec4 *hit)
{
#define ku modulo[acc->k+1]
#define kv modulo[acc->k+2]


    // not sure if this is best, might be better to set on struct construction?
    // otherwise, we're doing this for every triangle in the node, every time.
    // esp since we're visiting acc every time.
    // _could_ use the same struct for single rays if really needed.
    m128 n_u = _mm_set1_ps (acc->n_u);
    m128 n_v = _mm_set1_ps (acc->n_v);
    m128 n_d = _mm_set1_ps (acc->n_d);

    m128 b_nu = _mm_set1_ps (acc->b_nu);
    m128 b_nv = _mm_set1_ps (acc->b_nv);
    m128 b_d = _mm_set1_ps (acc->b_d);

    m128 c_nu = _mm_set1_ps (acc->c_nu);
    m128 c_nv = _mm_set1_ps (acc->c_nv);
    m128 c_d = _mm_set1_ps (acc->c_d);

    // Start high latency division early on - likely to get reordered?
    const m128 nd = _mm_set1_ps (1.f)/(r->dir.e[acc->k] + n_u*r->dir.e[ku] + n_v*r->dir.e[kv]);
    const m128 f = (n_d - r->orig.e[acc->k] - n_u*r->orig.e[ku] - n_v*r->orig.e[kv])*nd;

    // Check for valid distance
    m128 hitmask = _mm_and_ps((hit->dist > f), (f > _mm_set1_ps(EPSILON)));

    // Compute hit point positions on uv plane
    const m128 hu = (r->orig.e[ku] + f*r->dir.e[ku]);
    const m128 hv = (r->orig.e[kv] + f*r->dir.e[kv]);

    m128 zero = _mm_set1_ps(0);

    // Check first barycentric coordinate
    const m128 lambda = (hv*b_nu + hu*b_nv + b_d);
    hitmask = _mm_and_ps(hitmask, (lambda > zero));

    // Check second barycentric coordinate
    // TODO: (Kapsy) Replace these with FMAs.
    const m128 mue = (hu*c_nu + hv*c_nv + c_d);
    hitmask = _mm_and_ps(hitmask, (mue > zero));

    // Check third barycentric coordinate
    m128 one = _mm_set1_ps(1);
    hitmask = _mm_and_ps(hitmask, ((lambda + mue) <= one));

    // Have a valid hitpoint here, store it.
    hit->hitmask = hitmask;

    // hit->tri = index; // should already have this info????

    m128 hitmaskinv = _mm_xor_ps(hitmask, _mm_set1_epi32(0xffffffff));

    // NOTE: (Kapsy) Could _mm_or here but it doesn't matter that much.
    // Need to fine out which operation is actually faster.
    hit->dist = _mm_and_ps(hit->dist, hitmaskinv) + _mm_and_ps(f, hitmask);
    hit->u = _mm_and_ps(hit->u, hitmaskinv) + _mm_and_ps(lambda, hitmask);
    hit->v = _mm_and_ps(hit->v, hitmaskinv) + _mm_and_ps(mue, hitmask);

    hit->primref = _mm_and_ps(hit->primref, hitmaskinv) + _mm_and_ps(_mm_set1_ps(acc->index), hitmask);
    // up until now this didn't matter, as we checked the distance anyway.
    // but now we have triangle super imposed upon spheres, a mask must also be  applied to the primtype.
    hit->primtype = _mm_and_ps(hit->primtype, hitmaskinv) + _mm_and_ps(_mm_set1_ps(PrimTypeTri), hitmask);
}

  //////////////////////////////////////////////////////////////////////////////
 //// AA BSP Tree Functions ///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

inline void
PushStack (stack_item *bspstack, int &count, bsp_node *node, float tnear, float tfar)
{
    Assert (count < MAX_BSP_STACK_COUNT);
    stack_item *item = bspstack + count++;
    item->node = node;
    item->tnear = tnear;
    item->tfar = tfar;
}

// Should just macro these.
inline stack_item *
PopStack (stack_item *bspstack, int &count)
{
    stack_item *result = 0;

    Assert (count > 0);

    result = bspstack + count - 1;
    count--;

    return (result);
}

inline void
PushStack4 (stackitem4_t *bspstack4, int &count, bsp_node *node, m128 tnear, m128 tfar)
{
    Assert (count < MAX_BSP_STACK_COUNT);

    stackitem4_t *item = bspstack4 + count++;
    item->node = node;
    item->tnear = tnear;
    item->tfar = tfar;
}

// Should just macro these.
inline stackitem4_t *
PopStack4 (stackitem4_t *bspstack4, int &count)
{
    stackitem4_t *result = 0;

    Assert (count > 0);
    result = bspstack4 + count - 1;
    count--;

    return (result);
}


#define SIZEOF_BSP_NODE 8

inline bsp_node *
GetFrontChild (bsp_node *node, float dir)
{
    // NOTE: (Kapsy) Front -> back is lower -> higher when direction is positive.
    bsp_node *res = (bsp_node *)((char *)node +
                                 BSP_Offset(node) +
                                 (dir > 0.f ? 0 : SIZEOF_BSP_NODE));
    // need to check for

    // need to know which is front and which is back
    // sine of ray dir for the nodes dim should tell us?

    // handle degenerate cases here?

    return (res);
}

inline bsp_node *
GetBackChild (bsp_node *node, float dir)
{
    // NOTE: (Kapsy) Front -> back is lower -> higher when direction is positive.
    bsp_node *res = (bsp_node *)((char *)node +
                                 BSP_Offset(node) +
                                 (dir > 0.f ? SIZEOF_BSP_NODE : 0));
    return (res);
}

inline void
IntersectAllTrianglesInLeaf(bsp_node *node, ray *r, hitrec *hit)
{

    // NOTE: (Kapsy) Get the ref to the triaccs.
    // Need a count too.
    // Need to update the rays t?
    // Will just get the mesh for now anyway.

    int offset = BSP_Offset(node);

    if (offset == 0)
        return;

    triacclist_t *list = (triacclist_t *)((char *)node + offset);
    triacc *triaccbase = P32ToP (list->triaccs, triacc);

    for (int k=0 ; k<list->count ; k++)
    {
        triacc *acc = triaccbase + k;
        WaldIntersection(acc, r, hit);
    }
}

inline void
IntersectAllTrianglesInLeaf4 (bsp_node *node, ray4 *r, hitrec4 *hit)
{
    // NOTE: (Kapsy) Get the ref to the triaccs.
    // Need a count too.
    // Need to update the rays t?
    // Will just get the mesh for now anyway.

    int offset = BSP_Offset(node);

    if (offset == 0)
        return;

    triacclist_t *list = (triacclist_t *)((char *)node + offset);
    triacc *triaccbase = P32ToP (list->triaccs, triacc);

    for (int k=0 ; k<list->count ; k++)
    {
        triacc *acc = triaccbase + k;
        WaldIntersection4 (acc, r, hit);
    }
}


// NOTE: (Kapsy) Single ray implementation.
// Going to try this with a test ray and test bsp and see how it works...
// Interestingly, they say "object" so I guess we could go with object for now.
static void
TraverseTris (ray *r, rect3 *aabb, hitrec *hit, fastbsp_t *fastbsp)
{
    // NOTE: (Kapsy) Current stack items
    // Should make a pointer to the first node instead of doing this.
    bsp_node *node = P32ToP (fastbsp->nodes, bsp_node) + 1;

    float tnear = EPSILON;
    float tfar = MAXFLOAT; // should be thit, if we start using multiple boxes?

    // NOTE: (Kapsy) Clip the line segment to the bounding box.
    {
        float tmin;
        float tmax;
        float tymin;
        float tymax;
        float tzmin;
        float tzmax;

        v3 invdir = 1.f/r->dir;
        int signx = (invdir.x < 0);
        int signy = (invdir.y < 0);
        int signz = (invdir.z < 0);

        tmin = (aabb->e[signx].x - r->orig.x)*invdir.x;
        tmax = (aabb->e[1-signx].x - r->orig.x)*invdir.x;

        tymin = (aabb->e[signy].y - r->orig.y)*invdir.y;
        tymax = (aabb->e[1-signy].y - r->orig.y)*invdir.y;

        if ((tmin > tymax) || (tymin > tmax))
            return;
        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;

        tzmin = (aabb->e[signz].z - r->orig.z)*invdir.z;
        tzmax = (aabb->e[1-signz].z - r->orig.z)*invdir.z;

        if ((tmin > tzmax) || (tzmin > tmax))
            return;
        if (tzmin > tmin)
            tmin = tzmin;
        if (tzmax < tmax)
            tmax = tzmax;

        // We _should_ always have a collision with the box here, but need to check all cases.
        tnear = tmin;
        tfar = tmax;
    }

    if (tnear > tfar)
    {
        // NOTE: (Kapsy) Ray misses the bounding box.
        return;
    }

    int bspstackcount = 0;
    stack_item bspstack[MAX_BSP_STACK_COUNT];

    for(;;)
    {
        while (!BSP_IsLeaf(node))
        {
            int dim = BSP_Dim(node);

            // NOTE: (Kapsy) Keep traversing until the next leaf.
            // Pretty sure this means our dir has to be UNIT!!!
            float d = (node->inner.split - r->orig.e[dim])/r->dir.e[dim];

            if (d <= tnear)
            {
                // Case 1: d <= tnear <= tfar, therefore cull front side
                node = GetBackChild (node, r->dir.e[dim]);
            }
            else if (d >= tfar)
            {
                // Case 2: tnear <= tfar <= d, therefore cull back side
                node = GetFrontChild (node, r->dir.e[dim]);
            }
            else
            {
                // Case 3: Traverse both sides in turn
                bsp_node *backchild = GetBackChild (node, r->dir.e[dim]);
                PushStack (bspstack, bspstackcount, backchild, d, tfar);

                node = GetFrontChild (node, r->dir.e[dim]);
                tfar = d;
            }
        }

        // NOTE: (Kapsy) We now have a leaf.

        IntersectAllTrianglesInLeaf(node, r, hit);

        // NOTE: (Kapsy) Early ray termination.
        if (hit->dist <= tfar)
        {
            bspstackcount = 0;
            return;
        }

        if (bspstackcount == 0)
            return; // nothing left over to traverse

        stack_item *stack = PopStack (bspstack, bspstackcount);
        node = stack->node;
        tnear = stack->tnear;
        tfar = stack->tfar;
    }

}

// TODO: (Kapsy) Issue is that hit far is not being set when it should be.
static void
TraverseTris4 (ray4 *r, rect3 *aabb, hitrec4 *hit, m128 outmask, v3 signs, fastbsp_t *fastbsp)
{
    // NOTE: (Kapsy) Current stack items
    // Should make a pointer to the first node instead of doing this.
    // All of this stuff should be stored on the object, or the meta object struct.
    // bsp_node *node = fastbsp->nodes + 1;
    bsp_node *node = P32ToP (fastbsp->nodes, bsp_node) + 1;

    m128 tnear = _mm_set1_ps (EPSILON); // should be thit, if we start using multiple boxes?
    m128 tfar = _mm_set1_ps (MAXFLOAT);

    // TODO: (Kapsy) Confirm all rays have matching direction here?
    // For now will assume that all rays have the same direction signs at this point.
    // yeah, assume the check happens before here, and rays that don't comply are invalidated, and a single traversal is started.
    // Actually will need to ignore?

    // m128 outmask = _mm_set1_ps (0.f);
    // NOTE: (Kapsy) Clip the line segment to the bounding box.
    {
        m128 tmin;
        m128 tmax;
        m128 tymin;
        m128 tymax;
        m128 tzmin;
        m128 tzmax;

        // Cant do this.
        //// v3 invdir = 1.f/r->dir[0];
        //
        v34 invdir4 = _mm_set1_ps(1.f)/r->dir;

        // Just checking first for now, should assert if signs are different however.

        // This is breaking things.
        // Already have this info, need a way of passing.
        // int signx = (invdir4.x[0] < 0);
        // int signy = (invdir4.y[0] < 0);
        // int signz = (invdir4.z[0] < 0);

        int signx = (int)signs.x;
        int signy = (int)signs.y;
        int signz = (int)signs.z;

        tmin = (_mm_set1_ps (aabb->e[signx].x) - r->orig.x)*invdir4.x;
        tmax = (_mm_set1_ps (aabb->e[1-signx].x) - r->orig.x)*invdir4.x;

        tymin = (_mm_set1_ps (aabb->e[signy].y) - r->orig.y)*invdir4.y;
        tymax = (_mm_set1_ps (aabb->e[1-signy].y) - r->orig.y)*invdir4.y;


        //if ((tmin > tymax) || (tymin > tmax))
        //    return;
        outmask = _mm_or_ps(outmask, tmin > tymax);
        outmask = _mm_or_ps(outmask, tymin > tmax);

        ////if (AllBitsSet4 (outmask))
        ////    return;

        //  if (tymin > tmin)
        //      tmin = tymin;
        //  if (tymax < tmax)
        //      tmax = tymax;
        m128 yminmask = tymin > tmin;
        m128 yminmaskinv = _mm_xor_ps (yminmask, _mm_set1_epi32 (0xffffffff));
        tmin = _mm_or_ps (_mm_and_ps (tmin, yminmaskinv), _mm_and_ps (tymin, yminmask));

        m128 ymaxmask = tymax < tmax;
        m128 ymaxmaskinv = _mm_xor_ps (ymaxmask, _mm_set1_epi32 (0xffffffff));
        tmax = _mm_or_ps (_mm_and_ps (tmax, ymaxmaskinv), _mm_and_ps (tymax, ymaxmask));

        // todo:
        //// tmin = CompReplace (tymin > tmin, tmin, tymin);
        //// tmax = CompReplace (tymax > tmax, tmax, tymax);

        // tzmin = (aabb->e[signz].z - r->orig.z)*invdir.z;
        // tzmax = (aabb->e[1-signz].z - r->orig.z)*invdir.z;
        tzmin = (_mm_set1_ps (aabb->e[signz].z) - r->orig.z)*invdir4.z;
        tzmax = (_mm_set1_ps (aabb->e[1-signz].z) - r->orig.z)*invdir4.z;

        // if ((tmin > tzmax) || (tzmin > tmax))
        //     return;
        outmask = _mm_or_ps(outmask, tmin > tzmax);
        outmask = _mm_or_ps(outmask, tzmin > tmax);

        // if (tzmin > tmin)
        //     tmin = tzmin;
        // if (tzmax < tmax)
        //     tmax = tzmax;
        m128 zminmask = tzmin > tmin;
        m128 zminmaskinv = _mm_xor_ps (zminmask, _mm_set1_epi32 (0xffffffff));
        tmin = _mm_or_ps (_mm_and_ps (tmin, zminmaskinv), _mm_and_ps (tzmin, zminmask));

        m128 zmaxmask = tzmax < tmax;
        m128 zmaxmaskinv = _mm_xor_ps (zmaxmask, _mm_set1_epi32 (0xffffffff));
        tmax = _mm_or_ps (_mm_and_ps (tmax, zmaxmaskinv), _mm_and_ps (tzmax, zmaxmask));

        // We _should_ always have a collision with the box here, but need to check all cases.
        tnear = tmin;
        tfar = tmax;
    }

    if (AllBitsSet4 (outmask))
        return;

    // NOTE: (Kapsy) Make the ds for any outmask -1.
    //m128 invalidmask = _mm_or_ps ((tnear > tfar), outmask);
    //m128 d = _mm_and_ps (_mm_set1_ps (-1.f), invalidmask);

    // get the dim for the packet here.


    // NOTE: (Kapsy) Okay geddit now. all rays in the packet should be valid at this point.
    // Only when tnear < tfar, or they miss the bounding box entirely do they get invalidated.
    // From which point they stay that way for the remainder of the traversal only

    int bspstack4count = 0;
    stackitem4_t bspstack4[MAX_BSP_STACK_COUNT];

    for(;;)
    {
        while (!BSP_IsLeaf(node))
        {
            int dim = BSP_Dim(node);

            // Just using first element for now.
            // float dimdir = r->dir.e[dim][0];
            // This should work, but could make get back/front more efficient by using int???
            // What we actually want is int 1 for +ve and 0 for -ve
            float dimdir = -2*signs.e[dim] + 1;

            // NOTE: (Kapsy) Keep traversing until the next leaf.
            // Pretty sure this means our dir has to be UNIT!!!
            // float d = (node->inner.split - r->orig.e[dim])/r->dir.e[dim];

            m128 d = (_mm_set1_ps (node->inner.split) - r->orig.e[dim])/r->dir.e[dim];

            m128 active = (tnear < tfar);
            m128 activeinv = _mm_xor_ps (active, _mm_set1_epi32 (0xffffffff));

            if (AllBitsSet4 (_mm_or_ps((d <= tnear), activeinv)))
            //if (AllBitsSet4 (d <= tnear))
            {
                // Case 1: d <= tnear <= tfar for all active rays , therefore cull front side
                node = GetBackChild (node, dimdir);
            }
            else if (AllBitsSet4 (_mm_or_ps((d >= tfar), activeinv)))
            //else if (AllBitsSet4 (d >= tfar))
            {
                // Case 2: tnear <= tfar <= d for all active rays, therefore cull back side
                node = GetFrontChild (node, dimdir);
            }
            else
            {
                // Case 3: Traverse both sides in turn
                bsp_node *backchild = GetBackChild (node, dimdir);
                // These _shouldn't affect active, as tnear would be greater than far anyway.
                // So in the case where d is max, near will still be after far.
                PushStack4 (bspstack4, bspstack4count, backchild, Max4 (d, tnear), tfar);

                node = GetFrontChild (node, dimdir);

                // Likewise, if near is > far then that relationship won't change here.
                // Okay, so this is always going to set all to tfar???
                tfar = Min4 (d, tfar);
            }
        }

        // NOTE: (Kapsy) We now have a leaf.
        IntersectAllTrianglesInLeaf4 (node, r, hit);

        // NOTE: (Kapsy) Early ray termination.
        if (AllBitsSet4 (_mm_or_ps((hit->dist <= tfar), outmask)))
        // Early term is not the cause here.
        // if (AllBitsSet4 (hit->dist <= tfar))
        {
            bspstack4count = 0;
            return;
        }

        if (bspstack4count == 0)
            return; // nothing left over to traverse

        stackitem4_t *stack = PopStack4 (bspstack4, bspstack4count);
        node = stack->node;
        tnear = stack->tnear;
        tfar = stack->tfar;
    }

}


#if 0
static void
TraverseTest()
{
    // NOTE: (Kapsy) Setup a test bb
    rect3 aabb = { V3 (-1), V3 (1) };

    // NOTE: (Kapsy) Setup a test tree
    // NOTE: (Kapsy) In order to align node pairs to cacheline boundaries,
    // the root node must start at + 1.
    align_alloc bspalloc = AlignAlloc(sizeof(bsp_node)*MAX_BSP_NODES, CACHELINE_SIZE);
    bsptree = ((bsp_node *)bspalloc.start) + 1;
    bsp_node *bsptreeat = bsptree;

    // Inner node
    SetBSPType(bsptreeat, BSPTreeTypeInner);
    SetBSPDim(bsptreeat, DimX);
    int offset = ((char *)(bsptreeat + 1) - (char *)bsptreeat);
    SetBSPOffset(bsptreeat, offset);
    bsptreeat->inner.split = 0.0f;
    bsptreeat++;

    // First child
    SetBSPType (bsptreeat, BSPTreeTypeLeaf);
    SetBSPOffset (bsptreeat, 0);
    bsptreeat++;

    // Second child
    SetBSPType (bsptreeat, BSPTreeTypeLeaf);
    SetBSPOffset (bsptreeat, 0);
    bsptreeat++;

    // NOTE: (Kapsy) Should only hit front.
    // v3 O = V3 (-0.5, 0, 2.f);
    // v3 D = Unit (V3 (0, 0, -1.f));

    // NOTE: (Kapsy) Should hit the back and the back only.
    v3 O = V3 (0.5, 0, 2.f);
    v3 D = Unit (V3 (-0.3f, 0, -1.f));

    ray testray = Ray (O, D);

    Traverse(&testray, &aabb);
}
#endif

  //////////////////////////////////////////////////////////////////////////////
 //// Fast BSP Functions //////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void
DebugPrintFastBSP(fastbsp_t *fastbsp, int count)
{
    bsp_node *basenode = P32ToP (fastbsp->nodes, bsp_node);

    // Ignore the first.
    for (int i=1 ; i<count ; i++)
    {
        bsp_node *node = basenode + i;
        int type = BSP_IsLeaf(node) == 0 ? 0 : 1;

        float split = 0.f;

        if (type != NodeTypeLeaf)
            split = node->inner.split;

        int dim = BSP_Dim(node);
        int offset = BSP_Offset(node);

        printf("Node%.02d: t:%d d:%d os:%d s:%f\n", i, type, dim, offset, split);

        fflush(0);
    }
}

static bool
LoadFastBSP (fastbsp_t *fastbsp)
{
    bool result = false;

    FILE *file = fopen (BSP_FILENAME, "r");
    if (file)
    {
        size_t read = 0;

        fseek (file, 0, SEEK_SET);
        size_t filestart = ftell (file);
        fseek (file, 0, SEEK_END);
        size_t fileend = ftell (file);
        fseek (file, 0, SEEK_SET);

        size_t filesize = fileend - filestart;

        g_fastbsp = (fastbsp_t *) PoolAlloc (&g_bspmempool, filesize + CACHELINE_SIZE, CACHELINE_SIZE);
        read = fread ((void *) g_fastbsp, filesize, 1, file);

        fclose (file);

        result = true;

        printf ("Load fast BSP complete!\n");
    }

    return (result);
}

static void
SaveFastBSP (fastbsp_t *fastbsp)
{

    FILE *file = fopen (BSP_FILENAME, "w");
    if (file)
    {
        size_t written = 0;

        // NOTE: (Kapsy) What we're going to do now is just write the whole pool in as is.
        // The only thing to do is to fix up the pointers (fastbsp->triaccsbase etc)
        // Might be just as easy to make them relative.

        // Need to make sure that when we load back in that our pool is also cache size aligned.
        // This isn't perfect, but it should work.
        // Might want to assert on that!

        uint64_t poolsize = g_bspmempool.at - g_bspmempool.base;
        written = fwrite (g_bspmempool.base, poolsize, 1, file);

        fclose (file);
    }
}


  //////////////////////////////////////////////////////////////////////////////
 //// Make Node BSP ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct trilist_t
{
    int *indexes;
    int count;
    int pad;
};

struct makenode_t
{
    makenode_t *leftchild;
    makenode_t *rightchild;
    trilist_t trilist; // 32
    rect3 V; // 56
    int type;
    int depth; // temp until we figure out something nicer.
    int dim;
    float split; // 68 would be nice to get this into 64
};

#define MAX_BSP_DEPTH (50)
// NOTE: (Kapsy) Only alloc these during the setup phase.
// If we were really awesome, we could save the tree and only rebuild when then scene changes.
static makenode_t *g_makenodes;
static int g_makenodecount;

enum
{
    EdgeTypeStart,
    EdgeTypeEnd
};

struct edge_t
{
    float pos; // could use doubles here.
    int type;
    int triindex;
    int pad;
};

static int
EdgeCompare (const void *a, const void *b)
{
    edge_t *edgea = (edge_t *)a;
    edge_t *edgeb = (edge_t *)b;

    int res = -1;

    if (edgea->pos > edgeb->pos)
    {
        res = 1;
    }
    else if(edgea->pos == edgeb->pos)
    {
        res = (int) (edgea->type > edgeb->type);
    }

    return (res);
}

inline float
SurfaceArea (rect3 V)
{
    float Vw = V.max.x - V.min.x;
    float Vh = V.max.y - V.min.y;
    float Vd = V.max.z - V.min.z;

    float res = 2*(Vw*Vd + Vw*Vh + Vd*Vh);
    return (res);
}

inline void
DeleteDoubles(trilist_t *list)
{
    for (int i=0 ; i<list->count ; i++)
    {
        if (i+1 < list->count)
        {
            for (int j=i+1 ; j<list->count ; j++)
            {
                if (list->indexes[i] == list->indexes[j])
                {
                    // Swap in last and break;
                    list->indexes[j] = list->indexes[list->count - 1];
                    list->count--;
                    break;
                }
            }
        }
    }
}

#if 0
static void
MakeManualSplitTest(tri *tris, int tricount, makenode_t *node, int depth)
{
    rect3 V = node->V;

    int dim = 2;
    float split = 0.3f;

    rect3 VL = V;
    rect3 VR = V;

    int leftcount = tricount/2;
    int rightcount = tricount - leftcount;

    VL.max.e[dim] = split;
    VR.min.e[dim] = split;

    makenode_t *leftchild = g_makenodes + g_makenodecount++;
    leftchild->type = NodeTypeLeaf;
    leftchild->split = 0.f;
    leftchild->leftchild = 0;
    leftchild->rightchild = 0;
    leftchild->V = VL;

    leftchild->trilist.count = leftcount;
    leftchild->trilist.indexes = (int *)malloc(sizeof(int)*leftchild->trilist.count);
    for (int i=0 ; i<leftchild->trilist.count ; i++)
    {
        leftchild->trilist.indexes[i] = i;
    }

    makenode_t *rightchild = g_makenodes + g_makenodecount++;
    rightchild->type = NodeTypeLeaf;
    rightchild->split = 0.f;
    rightchild->rightchild = 0;
    rightchild->rightchild = 0;
    rightchild->V = VR;

    //rightchild->trilist.count = NV - costbestNL;
    rightchild->trilist.count = rightcount;
    rightchild->trilist.indexes = (int *)malloc(sizeof(int)*rightchild->trilist.count);
    for (int i=0 ; i<rightchild->trilist.count ; i++)
    {
        int triindex = i + leftcount;
        rightchild->trilist.indexes[i] = triindex;
    }

    node->type = NodeTypeInner;
    node->split = split;
    node->leftchild = leftchild;
    node->rightchild = rightchild;
    node->V = V;
    node->dim = dim;

    if (node->trilist.count)
    {
        node->trilist.count = 0;
        free (node->trilist.indexes);
    }

}
#endif

// NOTE: (Kapsy) This function recursively creates node splits, using the
// surface area heuristic described in Ingo Walds PHD.
// Recursively passes a list of triangles until a termination criteria is met.
// At that point we convert the "make" bsp tree into a "fast bsp tree, suitable
// for traveral.
#define MIN_TRI_COUNT 2


    static unsigned int g_mintricount = 0;
    static unsigned int g_maxbspdepth = 0;
    static unsigned int g_costbestfound = 0;

static void
MakeBSPNode(object_t *object, makenode_t *node, int depth)
{
    Assert (node->type == NodeTypeInner);

    int NV = node->trilist.count;
    rect3 V = node->V;

    // NOTE: (Kapsy) Find the dimension of maximum extent.
    v3 Vlen = V.max - V.min;
    float Vlenx = Vlen.x;
    float Vleny = Vlen.y;
    float Vlenz = Vlen.z;

    int dim = DimX;
    if ((Vleny >= Vlenx) && (Vleny >= Vlenz))
        dim = DimY;
    else if ((Vlenz >= Vlenx) && (Vlenz >= Vleny))
        dim = DimZ;

    int edgecount = node->trilist.count*2;
    edge_t *edges = (edge_t *)malloc (edgecount*sizeof(edge_t));

    // NOTE: (Kapsy) Add the edges for the axis.
    // List is just a list of tri indexes that we know are contained in the parent voxel, V.
    // Have to recalc the edges because the dim may be different.
    for (int i=0 ; i<node->trilist.count ; i++)
    {
        int triindex = node->trilist.indexes[i];

        tri_t *tri = object->tris + triindex;
        v3 A = object->verts[tri->A];
        v3 B = object->verts[tri->B];
        v3 C = object->verts[tri->C];

        float tridimmin = A.e[dim];
        float tridimmax = A.e[dim];

        if (B.e[dim] < tridimmin)
            tridimmin = B.e[dim];
        if (B.e[dim] > tridimmax)
            tridimmax = B.e[dim];

        if (C.e[dim] < tridimmin)
            tridimmin = C.e[dim];
        if (C.e[dim] > tridimmax)
            tridimmax = C.e[dim];

        edge_t *edgestart= edges + (i*2);
        edge_t *edgeend = edges + (i*2 + 1);

        edgestart->type = EdgeTypeStart;
        edgeend->type = EdgeTypeEnd;

        edgestart->triindex = triindex;
        edgeend->triindex = triindex;

        edgestart->pos = tridimmin;
        edgeend->pos = tridimmax;
    }

    // NOTE: (Kapsy) Sort the edges by pos.
    qsort (edges, edgecount, sizeof(edge_t), EdgeCompare);

    // NOTE: (Kapsy) Work out the lowest cost, as per Wald's PHD.
    // V: bounding box.
    // VL: left bounding box.
    // VR: right bounding box.

    // NOTE: (Kapsy) Cost of intersection, cost traversal.
    // float cisec = 1.f;
    // float ctrav = 3.f;

    float cisec = 1.f;
    float ctrav = 3.1f;

    float costbest = MAXFLOAT;
    int costbestidx = 0;
    int costbestNL = 0;
    int costbestNR = 0;

    // As we know
    int NL = 0;
    int NR = NV;

    float saV = SurfaceArea (V);
    float saVinv = 1.f/saV;

    int NLps = 0; // pending starts
        // NOTE: (Kapsy) Iterate all edges, find the best cost.
    for (int i=0 ; i<edgecount ; i++)
    {
        edge_t *e = edges + i;

        // TODO: (Kapsy) Think that this is a problem.
        // NL should only go up when we have an edge end right?
        //// if (e->type == EdgeTypeStart)
        ////     NL++;
        //// else if (e->type == EdgeTypeEnd)
        ////     NR--;

        // This seems to work, but really has no impact.
        // More pressingly, we're not getting enough empty nodes.
        if (e->type == EdgeTypeStart)
        {
            //NL++;
            NLps++;
        }
        else if (e->type == EdgeTypeEnd)
        {
            NR--;

            NL++;
            NLps--;
        }

        rect3 VL = V;
        rect3 VR = V;

        VL.max.e[dim] = e->pos;
        VR.min.e[dim] = e->pos;

        float saVL = SurfaceArea (VL);
        float saVR = SurfaceArea (VR);

        float pVLV = saVL*saVinv;
        float pVRV = saVR*saVinv;

        int NLc = NL + (NLps > 0 ? (NLps - 1) : 0);

        float costsplit = ctrav + cisec*(pVLV*NLc + pVRV*NR);

        // NOTE: (Kapsy) Surprisingly, this only makes minimal difference.
        if ((NLc == 0) || (NR == 0))
        {
            //costsplit *= 0.90f;
            costsplit *= 0.80f;
        }

        if (costsplit <= costbest)
        {
            costbest = costsplit;
            costbestidx = i;
            costbestNL = NLc;
            costbestNR = NR;
        }
    }

    float costleaf = NV*cisec;

    // NOTE: (Kapsy) Tallying stats.
    if (NV <= MIN_TRI_COUNT)
    {
        g_mintricount++;
    }
    else if (depth == MAX_BSP_DEPTH)
    {
        g_maxbspdepth++;
    }
    else if (costbest > costleaf)
    {
        g_costbestfound++;
    }

    // NOTE: (Kapsy) Check split criteria.
    if ((NV <= MIN_TRI_COUNT) || (depth == MAX_BSP_DEPTH) || (costbest > costleaf))
    {
        //printf("Created a leaf: NV:%d depth:%d\n", NV, depth);
        //fflush (0);

        // NOTE: (Kapsy) No point in splitting, we just make a leaf.
        node->type = NodeTypeLeaf;
        node->leftchild = 0;
        node->rightchild = 0;
        // NOTE: (Kapsy) node->tris and node->tricount stay the same
        return;
    }

    edge_t *beste = edges + costbestidx;

    rect3 VL = V;
    rect3 VR = V;
    VL.max.e[dim] = beste->pos;
    VR.min.e[dim] = beste->pos;

    // NOTE: (Kapsy) Create tri lists for either child, delete doubles.
    Assert (g_makenodecount < MAX_BSP_NODES);
    makenode_t *leftchild = g_makenodes + g_makenodecount++;
    leftchild->type = NodeTypeInner;
    leftchild->split = 0.f;
    leftchild->leftchild = 0;
    leftchild->rightchild = 0;
    leftchild->V = VL;

    leftchild->trilist.count = costbestidx;
    leftchild->trilist.indexes = (int *)malloc(sizeof(int)*leftchild->trilist.count);
    int *leftlistat = leftchild->trilist.indexes;
    for (int i=0 ; i<costbestidx ; i++)
    {
        edge_t *e = edges + i;
        *leftlistat = e->triindex;
        leftlistat++;
    }
    DeleteDoubles(&leftchild->trilist);

    Assert (g_makenodecount < MAX_BSP_NODES);
    makenode_t *rightchild = g_makenodes + g_makenodecount++;
    rightchild->type = NodeTypeInner;
    rightchild->split = 0.f;
    rightchild->rightchild = 0;
    rightchild->rightchild = 0;
    rightchild->V = VR;

    rightchild->trilist.count = edgecount - costbestidx;
    rightchild->trilist.indexes = (int *)malloc(sizeof(int)*rightchild->trilist.count);
    int *rightlistat = rightchild->trilist.indexes;
    for (int i=costbestidx ; i<edgecount ; i++)
    {
        edge_t *e = edges + i;
        *rightlistat = e->triindex;
        rightlistat++;
    }
    DeleteDoubles(&rightchild->trilist);

    // NOTE: (Kapsy) Setup the current node as inner.
    node->type = NodeTypeInner;
    node->split = beste->pos;
    node->leftchild = leftchild;
    node->rightchild = rightchild;
    node->V = V;
    node->dim = dim;

    // NOTE: (Kapsy) Once inner, remove the tri list for this node.
    if (node->trilist.count)
    {
        node->trilist.count = 0;
        free (node->trilist.indexes);
    }

    MakeBSPNode(object, leftchild, (depth + 1));
    MakeBSPNode(object, rightchild, (depth + 1));

    free (edges);
}

// NOTE: (Kapsy) Make to fast BSP conversion:
//
// Make structure:
//
//     A
//    / \
//   B   C
//  / \
// D   E
//
// And want to turn it into:
//
// order:   XABCDEFG
// offsets: 0123???? (? means we point to triangles)
//
// So in this case we would:
//
// (process A) (first node should already be added).
// add B, C
// set A offset to point to B
// push C to stack
//
// (process B)
// add D E
// set B offset to point to D
// push E to stack
//
// (process D)
// leaf so we create triaccs
// no more children, pop stack
//
// (process E)
// leaf so we create triaccs
// no more children, pop stack
//
// (process C)
// leaf so we create triaccs
// no more children, pop stack
//
// etc...

struct makenodepair_t
{
    makenode_t *makenode;
    bsp_node *fastnode;
};

static makenodepair_t *g_makenodepairs;
static int g_makenodepaircount;

inline makenodepair_t *
PopMakeNodePair()
{
    Assert (g_makenodepaircount);
    g_makenodepaircount--;
    makenodepair_t *result = g_makenodepairs + g_makenodepaircount;

    return (result);
}

inline void
PushMakeNodePair(makenode_t *makenode, bsp_node *fastnode)
{
    Assert(g_makenodepaircount < MAX_BSP_DEPTH);
    makenodepair_t *pair = g_makenodepairs + g_makenodepaircount++;

    pair->makenode = makenode;
    pair->fastnode = fastnode;

}


// NOTE: (Kapsy) Number of nodes.
static unsigned int g_bspstat_N = 0;
// NOTE: (Kapsy) Number of leaf nodes.
static unsigned int g_bspstat_NL = 0;
// NOTE: (Kapsy) Number of non empty leaf nodes.
static unsigned int g_bspstat_NNE = 0;
// NOTE: (Kapsy) Number of total triangles.
static unsigned int g_bspstat_NTT = 0;
// NOTE: (Kapsy) Number of leaf nodes with only one triangle.
static unsigned int g_bspstat_NOT = 0;
// NOTE: (Kapsy) Node count for depth.
static unsigned int g_bspstat_ND[MAX_BSP_DEPTH + 1] = {};

struct slownodestack_t
{
    unsigned int count;
    makenode_t nodes[MAX_BSP_STACK_COUNT];
};

inline void
PushSlowNode (slownodestack_t *stack, makenode_t *node)
{
    Assert (stack->count < MAX_BSP_STACK_COUNT);
    makenode_t *destnode = stack->nodes + stack->count++;
    // why can't we just use pointers?
    *destnode = *node;
}

inline makenode_t *
PopSlowNode (slownodestack_t *stack)
{
    Assert (stack->count > 0);
    makenode_t *node = stack->nodes + --stack->count;
    return (node);
}

// NOTE: (Kapsy) Okay, easiest thing for now is to store the depth with each node.
static void
CollateSlowBSPDebugInfo(makenode_t *node)
{
    node->depth = 0;

    unsigned int depth = 0;

    slownodestack_t stack = {};

    PushSlowNode(&stack, node);

    while (stack.count)
    {
        node = PopSlowNode(&stack);
        depth = node->depth;

        g_bspstat_N++;

        if (node->type == NodeTypeLeaf)
        {
            // printf ("Leaf: t:%d d:%d\n", node->trilist.count, depth);

            Assert (depth <= MAX_BSP_DEPTH);
            g_bspstat_ND[depth]++;

            g_bspstat_NL++;

            if (node->trilist.count)
            {
                g_bspstat_NNE++;
                g_bspstat_NTT += node->trilist.count;

                if (node->trilist.count == 1)
                {
                    g_bspstat_NOT++;
                }
            }

        }
        else if (node->type == NodeTypeInner)
        {
            // printf ("Inner: d: %d\n", depth);

            depth++;

            makenode_t *leftnode = node->leftchild;
            makenode_t *rightnode = node->rightchild;

            leftnode->depth = depth;
            rightnode->depth = depth;

            PushSlowNode(&stack, rightnode);
            PushSlowNode(&stack, leftnode);
        }
    }

    float NAT = (float) g_bspstat_NTT/(float) g_bspstat_NNE;

    printf ("SLOW BSP STATS:\n  Number of nodes: %d\n  Number of leaf nodes: %d\n  Number of non empty leaves: %d\n  Number of leaf nodes with one triangle: %d\n  Total triaccs: %d\n  Average triaccs per non empty leaf: %f\n", g_bspstat_N, g_bspstat_NL, g_bspstat_NNE, g_bspstat_NOT, g_bspstat_NTT, NAT);

    unsigned int total = 0;
    printf ("  Leaf nodes by depth:\n");
    for (int i=0 ; i<(MAX_BSP_DEPTH + 1) ; i++)
    {
        printf ("    %.02d: %d\n", i, g_bspstat_ND[i]);

        total+=g_bspstat_ND[i];
    }
    printf("total: %d\n", total);
}

struct createnodepair_t
{
    makenode_t *slownode;
    bsp_node *fastnode;
};

struct createnodepairstack_t
{
    unsigned int count;
    createnodepair_t pairs[MAX_BSP_STACK_COUNT];
};

inline void
PushCreateNodePair (createnodepairstack_t *stack, makenode_t *slownode, bsp_node *fastnode)
{
    Assert (stack->count < MAX_BSP_STACK_COUNT);
    createnodepair_t *pair = stack->pairs + stack->count++;

    pair->slownode = slownode;
    pair->fastnode = fastnode;
}

inline createnodepair_t *
PopCreateNodePair (createnodepairstack_t *stack)
{
    Assert (stack->count > 0);
    createnodepair_t *pair = stack->pairs + --stack->count;
    return (pair);
}

static void
CreateFastBSPFromSlow (fastbsp_t *fastbsp, object_t *object, makenode_t *slownode, bsp_node *fastnode)
{
    slownode->depth = 0;

    unsigned int depth = 0;

    createnodepairstack_t stack = {};

    PushCreateNodePair (&stack, slownode, fastnode);

    bsp_node *basenode = P32ToP (fastbsp->nodes, bsp_node);

    triacclist_t *triacclistbase = P32ToP (fastbsp->triacclist, triacclist_t);
    triacc *triaccsbase = P32ToP (fastbsp->triaccsbase, triacc);
    triacc *triaccat = triaccsbase;

    while (stack.count)
    {
        createnodepair_t *pair = PopCreateNodePair (&stack);

        slownode = pair->slownode;
        fastnode = pair->fastnode;

        depth = slownode->depth;

        g_bspstat_N++;

        if (slownode->type == NodeTypeLeaf)
        {
            Assert (depth <= MAX_BSP_DEPTH);

            // Assert (g_triacclistcount < ???);
            triacclist_t *list = triacclistbase + fastbsp->triacclistcount++;
            P32AssignP (list->triaccs, triaccat);
            list->count = 0;

            triacc *listtriaccat = triaccat;
            for (int i=0 ; i<slownode->trilist.count ; i++)
            {
                int triindex = slownode->trilist.indexes[i];
                CreateTriacc(object, triindex, listtriaccat);

                listtriaccat++;
                list->count++;
            }

            triaccat += list->count;
            Assert (triaccat <= (triaccsbase + fastbsp->triacccount));

            // triacc offset
            int64_t bspoffset = ((char *) list - (char *) fastnode);
            Assert (llabs(bspoffset) < (int)0x7fffffff);
            Assert ((bspoffset & 0x3) == 0);

            SetBSPOffset (fastnode, (int)bspoffset);
        }
        else if (slownode->type == NodeTypeInner)
        {
            depth++;

            // NOTE: (Kapsy) We need to add fast nodes contiguously, right after left.
            makenode_t *leftslownode = slownode->leftchild;
            makenode_t *rightslownode = slownode->rightchild;
            leftslownode->depth = depth;
            rightslownode->depth = depth;

            Assert (fastbsp->nodecount < MAX_BSP_NODES);
            bsp_node *leftfastnode = basenode + fastbsp->nodecount++;
            SetBSPType (leftfastnode, leftslownode->type);
            SetBSPDim (leftfastnode, leftslownode->dim);

            Assert (fastbsp->nodecount < MAX_BSP_NODES);
            bsp_node *rightfastnode = basenode + fastbsp->nodecount++;
            SetBSPType (rightfastnode, rightslownode->type);
            SetBSPDim (rightfastnode, rightslownode->dim);

            int leftfastchildoffset = (int)((char *)leftfastnode - (char *)fastnode);
            SetBSPOffset (fastnode, leftfastchildoffset);
            fastnode->inner.split = slownode->split;

            PushCreateNodePair (&stack, rightslownode, rightfastnode);
            PushCreateNodePair (&stack, leftslownode, leftfastnode);

        }
    }
}

