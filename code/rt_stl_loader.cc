
#pragma pack(push)
struct stl_binary_header
{
    char description[80];
    unsigned int facets;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct stl_facet
{
    v3 N;
    v3 e[3];
    unsigned short attributebytecount;
};
#pragma pack(pop)


static object_t *
LoadSTLObject (char *stlpath, m44 T)
{
    // NOTE: (Kapsy) Loading everything into g_object for now.
    object_t *object = &g_object;

    FILE *stlfile = fopen (stlpath, "r");
    Assert (stlfile);

    // NOTE: (Kapsy) Tri accs and mesh must come after nodes (for offsets).
    g_fastnodes = (bsp_node *) PoolAlloc (sizeof (bsp_node)*MAX_BSP_NODES, CACHELINE_SIZE);

    // NOTE: (Kapsy) Read in the header.
    stl_binary_header header = {};
    fread ((void *)&header, sizeof (header), 1, stlfile);

    // NOTE: (Kapsy) Read in the faces.
    int readsize = sizeof (stl_facet)*header.facets;
    stl_facet *facets = (stl_facet *)malloc (readsize);
    fread ((void *)facets, readsize, 1, stlfile);

    object->tricount = header.facets;
    object->tris = (tri_t *)malloc(sizeof(tri_t)*object->tricount);
    object->norms = (v3 *)malloc(sizeof(v3)*object->tricount);
    // NOTE: (Kapsy) Since we are an STL we just malloc for all 3 vertices.
    object->verts = (v3 *)malloc(sizeof(v3)*3*object->tricount);
    object->trimats = (int *)malloc(sizeof(int)*object->tricount);

    // NOTE: (Kapsy) Read in all facets, fill out triangles and verts.
    stl_facet *facetat = facets;
    tri_t *triat = object->tris;
    v3 *vertat = object->verts;
    v3 *normat = object->norms;
    int vertindexat = 0;
    int *trimatat = object->trimats;

    // TODO: (Kapsy) Should store on object.
    texture *ttest = &textures[texturecount++];
    ttest->type = TEX_PLAIN;
    ttest->albedo = V3 (1.f);

    object->mats = (mat_t *)malloc(sizeof(mat_t));
    mat_t *matat = object->mats;
    *matat = (mat_t) { MAT_NORMALS, ttest };

    rect3 aabb = {};

    for (int i=0 ; i<object->tricount ; i++)
    {
        facetat->e[0] = facetat->e[0]*T;
        facetat->e[1] = facetat->e[1]*T;
        facetat->e[2] = facetat->e[2]*T;

        *vertat++ = facetat->e[0];
        *vertat++ = facetat->e[1];
        *vertat++ = facetat->e[2];

        triat->A = vertindexat++;
        triat->B = vertindexat++;
        triat->C = vertindexat++;

        // Just the one mat for now.
        *trimatat++ = 0;

        for (int j=0 ; j<3 ; j++)
        {
            if(facetat->e[j].x > aabb.max.x)
                aabb.max.x = facetat->e[j].x;
            else if (facetat->e[j].x < aabb.min.x)
                aabb.min.x = facetat->e[j].x;

            if(facetat->e[j].y > aabb.max.y)
                aabb.max.y = facetat->e[j].y;
            else if (facetat->e[j].y < aabb.min.y)
                aabb.min.y = facetat->e[j].y;

            if(facetat->e[j].z > aabb.max.z)
                aabb.max.z = facetat->e[j].z;
            else if (facetat->e[j].z < aabb.min.z)
                aabb.min.z = facetat->e[j].z;
        }

        // NOTE: (Kapsy) Building up norms for now.

        v3 A = object->verts[triat->A];
        v3 B = object->verts[triat->B];
        v3 C = object->verts[triat->C];

        // Check order here! Should be right hand rule.
        v3 AB = B - A;
        v3 AC = C - A;
        v3 N = Unit (Cross (AB, AC));
        *normat++ = N;

        facetat++;
        triat++;
    }

    object->aabb = aabb;

    free (facets);
    fclose (stlfile);

    // CreateDebugBoundingBox (*aabb);

    return (object);
}
