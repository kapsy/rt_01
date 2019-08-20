enum
{
    TokenTypeHash,
    TokenTypeSlash,
    TokenTypeStreamEnd,

    // NOTE: (Kapsy) OBJ file tokens.
    TokenType_mtllib,
    TokenType_o,
    TokenType_v,
    TokenType_vn,
    TokenType_vt,
    TokenType_usemtl,
    TokenType_s,
    TokenType_f,

    // NOTE: (Kapsy) MTL file tokens.
    TokenType_newmtl,
    TokenType_Ns,
    TokenType_Ka,
    TokenType_Kd,
    TokenType_Ks,
    TokenType_Ke,
    TokenType_Ni,
    TokenType_d,
    TokenType_illum,
    TokenType_map_Kd,
    TokenType_disp,

    TokenTypeInt,
    TokenTypeFloat,
    TokenTypeElement,

    TokenTypePath,
};


struct tokendef_t
{
    int type;
    int count;
    char *chars;
};

static tokendef_t g_tokendefsobj[] = 
{
    { TokenTypeHash, 1, "#" },
    { TokenTypeSlash, 1, "/" },

    // NOTE: (Kapsy) Ordered longest to shortest.
    { TokenType_mtllib, 6, "mtllib" },
    { TokenType_usemtl, 6, "usemtl" },
    { TokenType_vn, 2, "vn" },
    { TokenType_vt, 2, "vt" },
    { TokenType_o, 1, "o" },
    { TokenType_v, 1, "v" },
    { TokenType_s, 1, "s" },
    { TokenType_f, 1, "f" },
};


struct tokenizer_t
{
    char *at;

    int tokendefcount;
    tokendef_t *tokendefs;
};

struct token_t
{
    int type;
    int count;
    char *chars;
};


struct triels_t
{
    int v, vt, vn;
};


inline void
SeekUntilWhiteSpace (tokenizer_t *tzer)
{
    while (!((*tzer->at == ' ') ||
             (*tzer->at == '\n') ||
             (*tzer->at == '\r')))
    {
        tzer->at++;
    }
}

inline void
SeekUntilNextLine (tokenizer_t *tzer)
{
    while (!(*tzer->at == '\n'))
    {
        tzer->at++;
    }

    // Jump over the newline char.
    // Should while over here?
    tzer->at++;
}

inline void
SkipWhiteSpace (tokenizer_t *tzer)
{
    while (((*tzer->at == ' ') ||
            (*tzer->at == '\n') ||
            (*tzer->at == '\r')))
    {
        tzer->at++;
    }
}


inline float
GetFloat (token_t token)
{
    char *s = (char *) malloc (token.count + 1);
    strncpy(s, token.chars, token.count);
    s[token.count] = '\0';

    float res = (float) atof (s);
    free (s);

    return (res);
}

inline int
GetInt (token_t token)
{
    char *s = (char *) malloc (token.count + 1);
    strncpy(s, token.chars, token.count);
    s[token.count] = '\0';

    int res = (int) atoi (s);
    free (s);

    return (res);
}

inline bool
Compare (token_t token, tokendef_t *tokendef)
{
    if (token.count != tokendef->count)
        return false;

    char *aat = token.chars;
    char *bat = tokendef->chars;

    while (*bat)
    {
        if (*bat++ != *aat++)
        {
            return false;
        }
    }

    return (true);
}

// NOTE: (Kapsy) This assumes we are already at a path, and just reads until white space.
static token_t
GetPathToken (tokenizer_t *tzer)
{
    token_t token = {};

    token.type = TokenTypePath;

    token.chars = tzer->at;

    SeekUntilWhiteSpace (tzer);

    token.count = tzer->at - token.chars;

    return (token);
}


static token_t
GetNextToken(tokenizer_t *tzer)
{
    token_t token = {};

    if (*tzer->at == '\0')
    {
        token.type = TokenTypeStreamEnd;
        return (token);
    }

    if (*tzer->at == '/')
    {
        token.type = TokenTypeSlash;
        token.count++;
        tzer->at++;
        return (token);
    }

    while (*tzer->at == '#')
    {
        SeekUntilNextLine(tzer);
    }

    // Seek up until next break - space, return, comment.
    token.chars = tzer->at;

    while (!((*tzer->at == ' ')  ||
             (*tzer->at == '\n') ||
             (*tzer->at == '\r') ||
             (*tzer->at == '/') ||
             (*tzer->at == '\0')))
    {
        token.count++;
        tzer->at++;
    }

    // This should skip comments too.
    SkipWhiteSpace (tzer);


    for (int i=0 ; i<tzer->tokendefcount ; i++)
    {
        // NOTE: (Kapsy) First look for predefined tokens.
        tokendef_t *tokendef = tzer->tokendefs + i;
        if (Compare (token, tokendef))
        {
            token.type = tokendef->type;
            return (token);
        }
    }

    // NOTE: (Kapsy) Then look for Floats and Ints.
    char numtest = *token.chars;
    if ((numtest == '-') ||
        ((numtest >= 0x30) && (numtest <= 0x39)))
    {
        token.type = TokenTypeInt;

        // NOTE: (Kapsy) Look for a decimal point.
        for (int i=0 ; i<token.count ; i++)
        {
            if (token.chars[i] == '.')
            {
                token.type = TokenTypeFloat;
                break;
            }
        }

        return (token);
    }

    // NOTE: (Kapsy) Then look for elements.
    token.type = TokenTypeElement;

    return (token);
}


inline v3
GetV3 (tokenizer_t *tzer)
{
    token_t token;
    v3 res = {};

    token = GetNextToken (tzer);
    Assert (token.type == TokenTypeFloat);
    res.x = GetFloat (token);

    token = GetNextToken (tzer);
    Assert (token.type == TokenTypeFloat);
    res.y = GetFloat (token);

    token = GetNextToken (tzer);
    Assert (token.type == TokenTypeFloat);
    res.z = GetFloat (token);

    return (res);
}

static int
GetMatIndexFromName (object_t *object, char *name)
{
    for (int i=0 ; i<object->matcount ; i++)
    {
        if (strcmp (object->mats[i].name, name) == 0)
        {
            return (i);
        }
    }

    return (-1);
}

static tokendef_t g_tokendefsmtl[] =
{
    { TokenTypeHash, 1, "#" },
    { TokenTypeSlash, 1, "/" },

    { TokenType_newmtl, 6, "newmtl" },
    { TokenType_Ns, 2, "Ns" },
    { TokenType_Ka, 2, "Ka" },
    { TokenType_Kd, 2, "Kd" },
    { TokenType_Ks, 2, "Ks" },
    { TokenType_Ke, 2, "Ke" },
    { TokenType_Ni, 2, "Ni" },
    { TokenType_d, 1, "d" },
    { TokenType_illum, 5, "illum" },
    { TokenType_map_Kd, 6, "map_Kd" },
    { TokenType_disp, 4, "disp" },
};

static void
LoadMTLMaterial (object_t *object, char *mtlpath)
{

    FILE *mtlfile = fopen (mtlpath, "r");
    Assert (mtlfile);

    int result;

    result = fseek (mtlfile, 0, SEEK_END);
    Assert (result == 0);
    size_t len = ftell (mtlfile);
    Assert (len >= 0);
    result = fseek (mtlfile, 0, SEEK_SET);
    Assert (result == 0);

    char *buf = (char *) malloc (len);
    fread ((void *) buf, len, 1, mtlfile);
    fclose (mtlfile);

      //////////////////////////////////////////////////////////////////////////
     //// Obtain Counts ///////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    int newmtlcount = 0;

    tokenizer_t _tzer = {};
    tokenizer_t *tzer = &_tzer;

    tzer->tokendefcount = sizeof (g_tokendefsmtl)/sizeof (g_tokendefsmtl[0]);
    tzer->tokendefs = g_tokendefsmtl;

    tzer->at = buf;
    token_t token;

    do
    {
        token = GetNextToken (tzer);
        switch (token.type)
        {
            case TokenType_newmtl:
                {
                    newmtlcount++;
                } break;
        }
    }
    while (token.type != TokenTypeStreamEnd);

      //////////////////////////////////////////////////////////////////////////
     //// Obtain Data /////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    object->matcount = newmtlcount;
    object->mats = (mat_t *)malloc(sizeof(mat_t)*object->matcount);
    mat_t *matat = object->mats;
    int matatindex = 0;

    tzer->at = buf;

    do
    {
        token = GetNextToken (tzer);

        switch (token.type)
        {
            case TokenType_newmtl:
                {
                    matat = object->mats + matatindex++;

                    // TODO: (Kapsy) Would rather follow the mtl way.
                    //
                    matat->tex = &textures[texturecount++];
                    matat->tex->type = TEX_PLAIN;
                    matat->tex->albedo = V3 (1.f);

                    matat->type = MAT_NORMALS;

                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeElement);

                    matat->name = (char *) malloc (token.count + 1);
                    strncpy(matat->name, token.chars, token.count);
                    matat->name[token.count] = '\0';

                } break;

            case TokenType_Ns:
                {
                    token = GetNextToken (tzer);
                    Assert ((token.type == TokenTypeFloat) || (token.type == TokenTypeInt));
                    matat->Ns = GetFloat (token);

                } break;

            case TokenType_Ka:
                {
                    matat->Ka = GetV3 (tzer);

                } break;

            case TokenType_Kd:
                {
                    matat->Kd = GetV3 (tzer);
                    matat->tex->albedo = matat->Kd;

                    //// static int testind = 0;
                    //// matat->tex->albedo = V3 (0);//matat->diffcol;
                    //// matat->tex->albedo.e[(testind++ % 3)] = 1.f;

                } break;

            case TokenType_Ks:
                {
                    matat->Ks = GetV3 (tzer);

                } break;

            case TokenType_Ke:
                {
                    matat->Ke = GetV3 (tzer);

                } break;

            case TokenType_Ni:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    matat->Ni = GetFloat (token);

                } break;

            case TokenType_d:
                {
                    token = GetNextToken (tzer);
                    //Assert (token.type == TokenTypeFloat);
                    Assert ((token.type == TokenTypeFloat) || (token.type == TokenTypeInt));
                    matat->d = GetFloat (token);

                } break;

            case TokenType_illum:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeInt);
                    matat->illum = GetInt (token);

                    matat->type = MAT_LAMBERTIAN;

                } break;

            case TokenType_map_Kd:
                {
                    // Fails b/c of absolute path.
                    token = GetPathToken (tzer);
                    Assert (token.type == TokenTypePath);

                    // NOTE: (Kapsy) Get C string from token.
                    int pathlen = token.count + 1;
                    char *mappath = (char *) malloc (pathlen);
                    strncpy(mappath, token.chars, pathlen);
                    mappath[pathlen - 1] = '\0';

                    // Make into texdiff.
                    matat->tex->type = TEX_BITMAP;
                    matat->tex->bufa.e =
                        stbi_load(mappath,
                            &matat->tex->bufa.w, &matat->tex->bufa.h,
                            &matat->tex->bufa.cpp, 3);

                    SeekUntilNextLine(tzer);

                } break;

            case TokenType_disp:
                {
                    token = GetPathToken (tzer);
                    Assert (token.type == TokenTypePath);

                    // NOTE: (Kapsy) Get C string from token.
                    int pathlen = token.count + 1;
                    char *mappath = (char *) malloc (pathlen);
                    strncpy(mappath, token.chars, pathlen);
                    mappath[pathlen - 1] = '\0';

                    printf("TokenType_disp: %s\n", mappath);

                    //matat->tex->type = TEX_BITMAP;
                    //matat->tex->bufa.e =
                    //    stbi_load(mappath,
                    //        &matat->tex->bufa.w, &matat->tex->bufa.h,
                    //        &matat->tex->bufa.cpp, 3);

                    // NOTE: (Kapsy) Since we don't support displacement maps, we convert them to normal maps.
                    {
                        texbuf_t buf = {};
                        buf.e = stbi_load(mappath, &buf.w, &buf.h, &buf.cpp, 3);

                        int w = buf.w;
                        int h = buf.h;
                        int cpp = buf.cpp;

                        // TODO: (Kapsy) Hmmm, is this anywhere to be found in the mat itself?
                        // Something wrong with our math if we have to bump this value by this much!?
                        // float a = 9.99f;
                        // float a = 3.99f;
                        float a = 3.1f;

                        // Hack to get lens sharper... Really not good to put here!
                        // MercMatHeadlampLensBump
                        if ((matatindex - 1) == 9)
                        {
                            a = 3.1f;
                        }

                        // Hack for the winker.
                        if ((matatindex - 1) == 20)
                        {
                            a = 9.f;
                        }

                        // Make this temp.
                        unsigned char *dest = (unsigned char *)malloc (sizeof (unsigned char)*cpp*w*h);

                        float dividend = 1.f/(float)0xff;

                        for (int v=1 ; v<(h - 1) ; v++)
                        {
                            for (int u=1 ; u<(w - 1) ; u++)
                            {
                                float ti0 = (float)*(buf.e + v*w*cpp + (u + 1)*cpp)*dividend;
                                float ti1 = (float)*(buf.e + v*w*cpp + (u - 1)*cpp)*dividend;

                                // From stb_image.h:
                                // The pixel data consists of *y scanlines of *x pixels,
                                // with each pixel consisting of N interleaved 8-bit components; the first
                                // pixel pointed to is top-left-most in the image. There is no padding between
                                // image scanlines or between pixels, regardless of format. The number of

                                // Need to find out which direction y the algo expects, vs what the jpeg is actually doing.
                                float tj0 = (float)*(buf.e + (v - 1)*w*cpp + u*cpp)*dividend;
                                float tj1 = (float)*(buf.e + (v + 1)*w*cpp + u*cpp)*dividend;

                                v3 S = V3 (1, 0, a*ti0 - a*ti1);
                                v3 T = V3 (0, 1, a*tj0 - a*tj1);

                                v3 SxT = Cross (S, T);
                                // v3 N = SxT/Length (SxT);
                                v3 N = V3 (-S.z, -T.z, 1.f)/sqrt (S.z*S.z + T.z*T.z + 1);

                                // NOTE: (Kapsy) Normalize for texture..
                                N.x = N.x*0.5f + 0.5f;
                                N.y = N.y*0.5f + 0.5f;
                                N.z = N.z*0.5f + 0.5f;

                                unsigned char *e = dest + v*w*cpp + u*cpp;

                                *(e + 0) = (unsigned char)(N.x*255.f);
                                *(e + 1) = (unsigned char)(N.y*255.f);
                                *(e + 2) = (unsigned char)(N.z*255.f);
                            }
                        }

                        texture *texnorm = &textures[texturecount++];
                        texnorm->bufa = buf;
                        texnorm->type = TEX_BITMAP;
                        texnorm->bufa.e = dest;

                        matat->texnorm = texnorm;
                    }

                } break;
        }
    }
    while (token.type != TokenTypeStreamEnd);
}

static triels_t GetTriElements (tokenizer_t *tzer)
{
    token_t token;

    triels_t res = {};

    token = GetNextToken (tzer);
    Assert (token.type == TokenTypeInt);
    res.v = GetInt (token);

    token = GetNextToken (tzer);
    Assert (token.type == TokenTypeSlash);

    token = GetNextToken (tzer);
    if (token.type == TokenTypeSlash)
    {
        res.vt = -1;
    }
    else if (token.type == TokenTypeInt)
    {
        res.vt = GetInt (token);
        token = GetNextToken (tzer);
        Assert (token.type == TokenTypeSlash);
    }

    token = GetNextToken (tzer);
    Assert (token.type == TokenTypeInt);
    res.vn = GetInt (token);

    return (res);
}

static object_t *
LoadOBJObject (char *objpath, m44 T)
{
    object_t *object = &g_object;

    FILE *objfile = fopen (objpath, "r");
    Assert (objfile);

    int result;

    result = fseek (objfile, 0, SEEK_END);
    Assert (result == 0);
    size_t len = ftell (objfile);
    Assert (len >= 0);
    result = fseek (objfile, 0, SEEK_SET);
    Assert (result == 0);

    char *buf = (char *) malloc (len);
    fread ((void *) buf, len, 1, objfile);
    fclose (objfile);

      //////////////////////////////////////////////////////////////////////////
     //// Obtain Counts ///////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    int ocount = 0;
    int vcount = 0;
    int vncount = 0;
    int vtcount = 0;
    int usemtlcount = 0;
    int fcount = 0;

    tokenizer_t _tzer = {};
    tokenizer_t *tzer = &_tzer;

    tzer->tokendefcount = sizeof (g_tokendefsobj)/sizeof (g_tokendefsobj[0]);
    tzer->tokendefs = g_tokendefsobj;

    tzer->at = buf;
    token_t token;

    do
    {
        token = GetNextToken (tzer);
        switch (token.type)
        {
            case TokenType_mtllib:
                {
                } break;

            case TokenType_o:
                {
                    ocount++;
                } break;

            case TokenType_v:
                {
                    vcount++;
                } break;

            case TokenType_vn:
                {
                    vncount++;
                } break;

            case TokenType_vt:
                {
                    vtcount++;
                } break;

            case TokenType_f:
                {
                    fcount++;
                } break;
        }
    }
    while (token.type != TokenTypeStreamEnd);

      //////////////////////////////////////////////////////////////////////////
     //// Obtain Data /////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    // Test stuff.
    texture *ttest = &textures[texturecount++];
    ttest->type = TEX_PLAIN;
    ttest->albedo = V3 (1.f);

    object->mats = (mat_t *)malloc(sizeof(mat_t));
    mat_t *matat = object->mats;
    *matat = (mat_t) { MAT_NORMALS, ttest };

    // Assumes all faces are tris.
    object->tricount = fcount;
    object->vertcount = vcount;
    object->vertnormcount = vncount;
    object->vertuvcount = vtcount;
    object->matcount = 0;

    // Rename to trivs or triverts.
    object->tris = (tri_t *)malloc(sizeof(tri_t)*object->tricount);
    object->trivns = (tri_t *)malloc(sizeof(tri_t)*object->tricount);
    object->trivts = (tri_t *)malloc(sizeof(tri_t)*object->tricount);
    object->trimats = (int *)malloc(sizeof(int)*object->tricount);

    object->norms = (v3 *)malloc(sizeof(v3)*object->tricount); // to delete

    object->verts = (v3 *)malloc(sizeof(v3)*object->vertcount);
    object->vertnorms = (v3 *)malloc(sizeof(v3)*object->vertnormcount); // not using these for anything for now.
    object->vertuvs = (v3 *)malloc(sizeof(v3)*object->vertuvcount); // not using these for anything for now.
    //object->mats = (mat_t *)malloc(sizeof(mat_t)*object->matcount);

    v3 *vertat = object->verts;
    v3 *vertnormat = object->vertnorms;
    v3 *vertuvat = object->vertuvs;

    tri_t *triat = object->tris;
    tri_t *trivnat = object->trivns;
    tri_t *trivtat = object->trivts;
    int *trimatat = object->trimats;

    int matindex = -1;

    tzer->at = buf;

    do
    {
        token = GetNextToken (tzer);

        switch (token.type)
        {
            case TokenType_mtllib:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeElement);

                    // NOTE: (Kapsy) Append the base obj path.
                    // TODO: (Kapsy) Really need to test if abs or rel. Can we?
                    int lastpathslash = 0;
                    for (int i=0 ; i<strlen (objpath) ; i++)
                    {
                        if (objpath[i] == '/')
                            lastpathslash = i;
                    }

                    int pathlen = lastpathslash + 1;
                    int filelen = token.count + 1;
                    char *mtlpath = (char *) malloc (pathlen + filelen);
                    strncpy(mtlpath, objpath, pathlen);
                    strncpy(mtlpath + pathlen, token.chars, token.count);
                    mtlpath[pathlen + filelen - 1] = '\0';

                    LoadMTLMaterial (object, mtlpath);

                    free (mtlpath);

                } break;

            case TokenType_o:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeElement);

                    // NOTE: (Kapsy) We ignore objects for now.

                } break;

            case TokenType_v:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float a = GetFloat (token);

                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float b = GetFloat (token);

                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float c = GetFloat (token);

                    *vertat++ = V3 (a, b, c)*T;
                    //*vertat++ = V3 (a, b, c);

                } break;

            case TokenType_vn:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float a = GetFloat (token);

                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float b = GetFloat (token);

                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float c = GetFloat (token);

                    // This is wrong, all vns are local to the vertex itself, so we only want to apply rotations to them
                    //*vertnormat++ = V3 (a, b, c)*T;
                    *vertnormat++ = V3 (a, b, c);

                } break;

            case TokenType_vt:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float u = GetFloat (token);

                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeFloat);
                    float v = GetFloat (token);

                    *vertuvat++ = V3 (u, v, 0.f);

                } break;

            case TokenType_usemtl:
                {
                    token = GetNextToken (tzer);
                    Assert (token.type == TokenTypeElement);

                    char *name = (char *) malloc (token.count + 1);
                    strncpy(name, token.chars, token.count);
                    name[token.count] = '\0';

                    matindex = GetMatIndexFromName (object, name);
                    Assert (matindex >= 0);

                    free (name);

                } break;

            case TokenType_s:
                {
                    // NOTE: (Kapsy) We ignore smoothing for now.
                    token = GetNextToken (tzer);
                    Assert ((token.type == TokenTypeElement) || (token.type == TokenTypeInt));

                } break;

            case TokenType_f:
                {
                    // NOTE: (Kapsy) We assume all faces are tris!
                    triels_t elA = GetTriElements (tzer);
                    triels_t elB = GetTriElements (tzer);
                    triels_t elC = GetTriElements (tzer);

                    // NOTE: (Kapsy) Assign vertex indexes.
                    triat->A = elA.v - 1;
                    triat->B = elB.v - 1;
                    triat->C = elC.v - 1;
                    triat++;

                    // TODO: (Kapsy) Use real mats
                    *trimatat++ = matindex;

                    // NOTE: (Kapsy) Assign vertex normal indexes.
                    // might want to split these up. will leave for now.
                    trivnat->A = elA.vn - 1;
                    trivnat->B = elB.vn - 1;
                    trivnat->C = elC.vn - 1;
                    trivnat++;


                    mat_t *matcheck = object->mats + matindex;
                    if (matcheck->texnorm)
                    {
                        Assert (elA.vt >= 0);
                        Assert (elB.vt >= 0);
                        Assert (elC.vt >= 0);
                    }

                    trivtat->A = elA.vt - 1;
                    trivtat->B = elB.vt - 1;
                    trivtat->C = elC.vt - 1;
                    trivtat++;

                } break;
        }
    }
    while (token.type != TokenTypeStreamEnd);


    rect3 aabb = {};
    float epsilon = 1.f;
    v3 *normat = object->norms;
    for (int i=0 ; i<object->tricount ; i++)
    {
        tri_t *tri = object->tris + i;

        for (int j=0 ; j<3 ; j++)
        {
            v3 v = *(object->verts + tri->e[j]);

            if(v.x > aabb.max.x)
                aabb.max.x = v.x + epsilon;
            else if (v.x < aabb.min.x)
                aabb.min.x = v.x - epsilon;

            if(v.y > aabb.max.y)
                aabb.max.y = v.y + epsilon;
            else if (v.y < aabb.min.y)
                aabb.min.y = v.y - epsilon;

            if(v.z > aabb.max.z)
                aabb.max.z = v.z + epsilon;
            else if (v.z < aabb.min.z)
                aabb.min.z = v.z - epsilon;
        }


        // Not using norms
        //// tri_t *trivn = object->trivns + i;
        //// v3 A = object->vertnorms[trivn->A];
        //// v3 B = object->vertnorms[trivn->B];
        //// v3 C = object->vertnorms[trivn->C];

        //// // Check order here! Should be right hand rule.
        //// //v3 AB = B - A;
        //// //v3 AC = C - A;
        //// //v3 N = Unit (Cross (AB, AC));
        //// //v3 N = Unit (Cross (AC, AB));
        //// *normat++ = N;
    }

    object->aabb = aabb;

    free (buf);

    fclose (objfile);

    // CreateDebugBoundingBox (*aabb);

    return (object);
}

