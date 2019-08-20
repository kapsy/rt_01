
#if 0

    for (int e=0 ; e<4 ; e++)
    {
        v3 res = {};

        // janky shit to get it going
        hitrec hit = {};
        hit.dist = hit4.dist[e];
        hit.u = hit4.u[e];
        hit.v = hit4.v[e];
        hit.primref = hit4.primref[e];
        hit.primtype = hit4.primtype[e];


        // janky shit to get it going
        ray _r = {};
        _r.orig.x = r4->orig.x[e];
        _r.orig.y = r4->orig.y[e];
        _r.orig.z = r4->orig.z[e];
        _r.dir.x = r4->dir.x[e];
        _r.dir.y = r4->dir.y[e];
        _r.dir.z = r4->dir.z[e];
        ray *r = &_r;

        //TraverseTris (r, &object->aabb, &hit);

        if (hit.dist < MAXFLOAT)
        {
            //res = V3 (0,0,1)*hit.dist;

            mat_t *mat = 0;
            int matindex = 0;
            v3 N;
            v3 N1;
            v3 p;

            float w, u, v;

            switch(hit.primtype)
            {
                case PrimTypeTri:
                    {
                        int triindex = hit.primref;
                        //tri_t *tri = object->tris + triindex;
                        mat = object->mats + object->trimats[triindex];
                        matindex = object->trimats[triindex];

                        // would need to get out vert normals if smooth shading...

                        // should rename dist to t
                        //N = object->norms[triindex];

                        {
                            tri_t *tri = object->tris + triindex;
                            v3 A = object->verts[tri->A];
                            v3 B = object->verts[tri->B];
                            v3 C = object->verts[tri->C];
                            v3 AB = B - A;
                            v3 AC = C - A;
                            N1 = Unit (Cross (AB, AC));
                        }

                        u = hit.u;
                        v = hit.v;
                        w = (1.f - u - v);

                        tri_t *trivn = object->trivns + triindex;
                        v3 A = object->vertnorms[trivn->A];
                        v3 B = object->vertnorms[trivn->B];
                        v3 C = object->vertnorms[trivn->C];
                        N = Unit (w*A + u*B + v*C);
                        // Okay here's my take on this whole lerp thing
                        // I don't think that it's so much that there's a prob with the lerp per se, or even that the reflection code requires something special
                        // Think it's much more likely that our vns are just pointing in slightly the wrong direction. That's my take. How they became like that I don't know.
                        // Almost certain that this is it because the norms on the bonnet for the vn version are the same color as for the rear of the car for the N version
                        // Best way to test this is to calculate the vertex norms ourselves.
                        // N = Unit ((A + B + C)/3.f);
                        // Other possibilities:
                        // X Using wrong wuv coords, although this seems unlikely, smooth shading would be broken if so.
                        // precision error from rotating and scaling. Although this is also hard to consider, the whole model would be warped if so.
                        // Blender exporting wrong. This is possible. Need to build from scratch to find out.
                        //     This was it. Not rotating fixed the problem.
                        // Reading in wrong. Hard to imagine, but possible.

                        p = r->orig + (hit.dist)*r->dir;

                    } break;

                case PrimTypeSphere:
                    {
                        int sphereindex = hit.primref;

                        float rad = spheres[sphereindex].rad;
                        v3 center = spheres[sphereindex].center;

                        p = PointAt (r, hit.dist);
                        N = (p - center)/rad;
                        mat = &spheres[sphereindex].mat;
                        matindex = EarthTexture;

                    } break;
            }

            // testing.
            res = 0.5*V3 (N.x+1, N.y+1, N.z+1);

matindex = -1;

            //res = wuv;

            switch (matindex)
            {
                case MercMatBlackTrim:
                    {
                        res = V3 (0.1);//V3 (0.35);
                    } break;

                case MercMatBlackTrim2:
                    {
                        res = 0.5*V3 (N.x+1, N.y+1, N.z+1);
                    } break;

                case MercMatBody:
                    {
                        //v3 albedo = V3 (0.6, 0.6, 1.f);
                        v3 albedo = V3 (1.f);

#if 1
                        v3 ref = V3 (0.f);
                        // So plan is to clamp the reflected vector:
                        {
                            v3 v = Unit (r->dir);
                            v3 reflected = v - 2*Dot (v, N)*N;
                            v3 bias = N*1e-4;
                            ray scattered = Ray (p + bias, reflected);// + mat->fuzz*RandomInUnitSphere ());


                            if (depth < MAX_DEPTH)
                            {
                                bool result = (Dot (scattered.dir, N) > 0.0f);
                                if(result)
                                {
                                    //res = attenuation*GetColorForRay (&scattered, object, depth+1);
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                }
                                else
                                {
                                    // this only makes a minimal difference for 1M tris.
                                    // would be useful for pure reflective objects, but prob not worth worrying about for most.
                                    v3 bias = N1*(1e-4 + 0.01);

                                    reflected = v - Dot (v, N1)*N1;
                                    scattered = Ray (p+bias, reflected+bias);// + mat->fuzz*RandomInUnitSphere ());
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                    //// res = V3 (0);

                                    //// reflected = v - 2*Dot (v, N1)*N1;
                                    //// bias = N1*1e-4;
                                    //// scattered = Ray (p + bias, reflected + bias);// + mat->fuzz*RandomInUnitSphere ());
                                    //// res = GetColorForRay (&scattered, object, depth+1);

                                    // res = V3(0,1,0);
                                }
                            }
                        }


#endif

#define ApplyGlobalIllum(base, change, illum) ((base)*illum*g_illumcol + (change)*(1.f - illum))

                        v3 L = -g_light.dir;

                        res = ApplyGlobalIllum (albedo, albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L)), g_illum);

                        float bias = 1e-4;
                        ray sray = {};
                        sray.orig = p + N*bias;
                        sray.dir = L;

                        hitrec shit = {};
                        shit.dist = MAXFLOAT;
                        TraverseTris (&sray, &object->aabb, &shit);

                        if (shit.dist < MAXFLOAT)
                        {
                            res = ApplyGlobalIllum (res, res*0.5, g_illum*0.7);
                        }
                        else
                        {
                            res= res;
                        }

                        res = res*0.4 + 0.6*res*ref;

                        // This isn't working at all...
                        ////        v3 diff = {};
                        ////        v3 spec = {};
                        ////        {
                        ////
                        ////        pointlight_t *light = g_pointlights + 0;
                        ////
                        ////        float bias = 1e-4;
                        ////        v3 lightdir = g_light.dir;
                        ////        float r2 = Dot (lightdir, lightdir);
                        ////        float dist = sqrtf(r2);
                        ////        lightdir/=dist;
                        ////        v3 intensity = g_light.intensity*g_light.col/(4.f*M_PI*r2);
                        ////
                        ////        diff += albedo*intensity*Clamp01 (Dot (N, lightdir));
                        ////
                        ////        v3 I = lightdir;
                        ////        v3 R = I - 2*Dot (I, N)*N;
                        ////
                        ////        float n = 1500;
                        ////        float Kd = 0.3;
                        ////        float Ks = 0.02;
                        ////
                        ////        spec += intensity*pow (Clamp01 (Dot (R, r->dir)), n);
                        ////
                        ////        res += diff*mat->Kd + spec*mat->Ks;
                        ////        }

                    } break;

                case MercMatChrome:
                    {
                        res = V3 (0,0,1);
                    } break;

                case MercMatChromeTrim:
                    {
                        v3 albedo = V3 (1,1,0);

                        v3 L = -g_light.dir;
                        res = ApplyGlobalIllum (albedo, albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L)), g_illum);

                        float bias = 1e-4;
                        ray sray = {};
                        sray.orig = p + N*bias;
                        sray.dir = L;

                        hitrec shit = {};
                        shit.dist = MAXFLOAT;
                        TraverseTris (&sray, &object->aabb, &shit);

                        if (shit.dist < MAXFLOAT)
                        {
                            res = ApplyGlobalIllum (res, res*0.5, g_illum);
                        }
                        else
                        {
                            res= res;
                        }

                        v3 ref = V3 (0.f);
                        {
                            v3 v = Unit (r->dir);
                            v3 reflected = v - 2*Dot (v, N)*N;
                            v3 bias = N*1e-4;
                            ray scattered = Ray (p + bias, reflected);// + mat->fuzz*RandomInUnitSphere ());

                            if (depth < MAX_DEPTH)
                            {
                                bool result = (Dot (scattered.dir, N) > 0.0f);
                                if(result)
                                {
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                }
                                else
                                {
                                    // this only makes a minimal difference for 1M tris.
                                    // would be useful for pure reflective objects, but prob not worth worrying about for most.
                                    v3 bias = N1*(1e-4 + 0.01);

                                    reflected = v - Dot (v, N1)*N1;// - bias*30.f;
                                    scattered = Ray (p+bias, reflected+bias);// + mat->fuzz*RandomInUnitSphere ());
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                }
                            }
                        }

                        res = ref*0.98;



                    } break;

                case MercMatDarkChrome:
                    {
                        res = V3 (0.03);
                    } break;

                case MercMatGauges:
                    {
                        res = V3 (1,1,1);
                    } break;

                case MercMatGlass:
                    {
                        // Obtain kr, reflection/refraction ratio.
                        float kr = 0.f;
                        {

                            v3 bias = 1e-4*N;

                            float ior = 1.4;

                            v3 I = Unit (r->dir);
                            float cosi = Clamp (-1.f, 1.f, Dot (I, N));

                            v3 n = N;

                            // NOTE: (Kapsy) Index of refraction before entering the medium (air).
                            float etai = 1;
                            // NOTE: (Kapsy) Refraction index of the object that the ray has hit.
                            float etat = ior;

                            bool outside = cosi < 0.f;

                            if (outside)
                            {
                                // Outside the surface.
                                cosi = -cosi;
                            }
                            else
                            {
                                // Inside the surface, need to reverse the normal.
                                //n = -N;
                                // Not so true for glass - no need to swap?
                                //Swap (etai, etat);
                                cosi = Clamp (-1.f, 1.f, Dot (I, -n));
                                cosi = -cosi;
                            }


                            // NOTE: (Kapsy) Check for total internal reflection.
                            float sint = etai/etat*sqrt (max (0.f, 1.f - cosi*cosi));

                            if (sint >= 1)
                            {
                                kr = 1.f;
                            }
                            else
                            {
                                float cost = sqrt (max (0.f, 1.f - sint*sint));
                                cosi = fabs (cosi);

                                //// float Rs = ((etat*cosi) - (etai*cost))/((etat*cosi) + (etai*cost));
                                //// float Rp = ((etai*cosi) - (etat*cost))/((etai*cosi) + (etat*cost));
                                //// kr = (Rs*Rs + Rp*Rp)*0.5f;

                                float n1 = etai;
                                float n2 = etat;

                                float cos1 = fabsf (cosi);
                                float cos2 = sqrt (max (0.f, 1.f - sint*sint));

                                float Fr1 = (n2*cos1 - n1*cos2)/(n2*cos1 + n1*cos2);
                                float Fr2 = (n1*cos2 - n2*cos1)/(n1*cos2 + n2*cos1);

                                float Fr = (Fr1*Fr1 + Fr2*Fr2)*0.5f;
                                kr = Fr;
                            }

                        }

                        //res = V3 (0, kr, 0);
                        v3 refractcol = V3 (1.f);
                        {
                            v3 bias = 1e-4*N;

                            float ior = 1.4;

                            v3 I = (r->dir);
                            float cosi = Clamp (-1.f, 1.f, Dot (I, N));

                            // NOTE: (Kapsy) Index of refraction before entering the medium (air).
                            float etai = 1;
                            // NOTE: (Kapsy) Refraction index of the object that the ray has hit.
                            float etat = ior;

                            bool outside = cosi < 0.f;

                            // NOTE: (Kapsy) Work out refraction if not total internal reflection.
                            if (kr < 1.f)
                            {
                                // Obtain refraction direction.
                                v3 n = N;

                                if (outside)
                                {
                                    // Outside the surface.
                                    cosi = -cosi;
                                }
                                else
                                {
                                    // Inside the surface, need to reverse the normal.
                                    n = -N;
                                    // Not so true for glass - no need to swap?
                                    Swap (etai, etat);
                                }

                                float eta = etai/etat;
                                float k = 1 - eta*eta*(1 - cosi*cosi);
                                //Assert (k >= 0.f);

                                if ((k > 0.f) && (depth < MAX_DEPTH))
                                {
                                    v3 refractdir = eta*I + (eta*cosi - sqrtf (k))*n;
                                    v3 refractorig = outside ? p - bias : p + bias;
                                    ray refractray = Ray (refractorig, refractdir);
                                    refractcol = GetColorForRay (&refractray, object, depth+1);
                                }
                                else
                                {
                                    refractcol = V3 (1,0,0);
                                }
                            }

                            res = refractcol;


                            v3 reflectorig = outside ? p + bias : p - bias;
                            v3 reflectdir = Reflect (I, N);
                            ray reflectray = Ray (reflectorig, reflectdir);
                            v3 reflectcol = V3 (1,0.5,1);

                            if (depth < MAX_DEPTH)
                                reflectcol = GetColorForRay (&reflectray, object, depth+1);

                            if (depth < MAX_DEPTH)
                            {
                                //res = V3 (0.75)*(reflectcol*kr + refractcol*(1.f - kr));
                                //kr = kr/kr*5.5f;
                                kr = Clamp01 (kr*9.f);
                                res = V3(0.6, 0.9, 1)*((reflectcol)*kr + refractcol*(1.f - kr));
                            }
                            else
                            {
                                res = V3 (0.f);
                            }
                        }

                    } break;


                case MercMatLights:
                    {
                        v3 ref = V3 (0.f);
                        // So plan is to clamp the reflected vector:
                        {
                            v3 v = Unit (r->dir);
                            v3 reflected = v - 2*Dot (v, N)*N;
                            v3 bias = N*1e-4;
                            ray scattered = Ray (p + bias, reflected);// + mat->fuzz*RandomInUnitSphere ());

                            if (depth < MAX_DEPTH)
                            {
                                bool result = (Dot (scattered.dir, N) > 0.0f);
                                if(result)
                                {
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                }
                            }
                        }

                        res = V3 (0.2f) + ref*0.3f;

                    } break;

                case MercMatLogo:
                    {
                        res = V3 (1,0,0);
                    } break;

                case MercMatMirror:
                    {
                        v3 ref = V3 (0.f);
                        // So plan is to clamp the reflected vector:
                        {
                            v3 v = Unit (r->dir);
                            v3 reflected = v - 2*Dot (v, N)*N;
                            v3 bias = N*1e-4;
                            ray scattered = Ray (p + bias, reflected);// + mat->fuzz*RandomInUnitSphere ());

                            if (depth < MAX_DEPTH)
                            {
                                bool result = (Dot (scattered.dir, N) > 0.0f);
                                if(result)
                                {
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                }
                            }
                        }

                        res = ref;

                    } break;

                case MercMatRed_Carpet:
                case MercMatRedLeather:
                    {

                        float dirdotN = Dot (Unit (r->dir), N);

                        if (dirdotN < 0.f)
                        {

                            v3 albedo = V3 (1,0,0);

                            v3 L = -g_light.dir;
                            res = ApplyGlobalIllum (albedo, albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L)), g_illum);

                            float bias = 1e-4;
                            ray sray = {};
                            sray.orig = p + N*bias;
                            sray.dir = L;

                            hitrec shit = {};
                            shit.dist = MAXFLOAT;
                            TraverseTris (&sray, &object->aabb, &shit);

                            if (shit.dist < MAXFLOAT)
                            {
                                res = ApplyGlobalIllum (res, res*0.5, g_illum);
                            }
                            else
                            {
                                res= res;
                            }
                        }
                        else
                        {
                            res = V3 (0.f);
                        }


                    } break;

                case MercMatRubber:
                    {
                        v3 albedo = V3 (0.15);

                        v3 L = -g_light.dir;
                        res = ApplyGlobalIllum (albedo, albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L)), g_illum);

                        float bias = 1e-4;
                        ray sray = {};
                        sray.orig = p + N*bias;
                        sray.dir = L;

                        hitrec shit = {};
                        shit.dist = MAXFLOAT;
                        TraverseTris (&sray, &object->aabb, &shit);

                        if (shit.dist < MAXFLOAT)
                        {
                            res = ApplyGlobalIllum (res, res*0.5, g_illum);
                        }
                        else
                        {
                            res= res;
                        }

                    } break;

                case MercMatWhiteTrim:
                    {
                        v3 albedo = V3 (1);

                        v3 L = -g_light.dir;
                        res = ApplyGlobalIllum (albedo, albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L)), g_illum);

                        float bias = 1e-4;
                        ray sray = {};
                        sray.orig = p + N*bias;
                        sray.dir = L;

                        hitrec shit = {};
                        shit.dist = MAXFLOAT;
                        TraverseTris (&sray, &object->aabb, &shit);

                        if (shit.dist < MAXFLOAT)
                        {
                            res = ApplyGlobalIllum (res, res*0.5, g_illum);
                        }
                        else
                        {
                            res= res;
                        }
                    } break;

                case EarthTexture:
                    {
                        v3 rand = RandomInUnitSphere ();

                        v3 target = p + N + rand;

                        ray scattered = Ray (p, target - p);

                        Assert (mat->tex);
                        v3 albedo = V3 (0.94,1,0.8)*GetAttenuation (mat->tex, 0, 0, p);
                        albedo = albedo*0.85;

#if 1
                        v3 L = -g_light.dir;
                        res = ApplyGlobalIllum (albedo, albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L)), g_illum);

                        float bias = 1e-4;
                        ray sray = {};
                        sray.orig = p + N*bias;
                        sray.dir = L;

                        hitrec shit = {};
                        shit.dist = MAXFLOAT;
                        TraverseTris (&sray, &object->aabb, &shit);

                        if (shit.dist < MAXFLOAT)
                        {
                            res = ApplyGlobalIllum (res, res*0.5, g_illum);
                        }
                        else
                        {
                            res= res;
                        }


                        v3 ref = V3 (0.f);
                        // So plan is to clamp the reflected vector:
                        {
                            v3 v = Unit (r->dir);
                            v3 reflected = v - 2*Dot (v, N)*N;
                            v3 bias = N*1e-4;
                            ray scattered = Ray (p + bias, reflected);// + mat->fuzz*RandomInUnitSphere ());

                            if (depth < MAX_DEPTH)
                            {
                                bool result = (Dot (scattered.dir, N) > 0.0f);
                                if(result)
                                {
                                    ref = GetColorForRay (&scattered, object, depth+1);
                                }
                            }
                        }

                        res = res*ref;
#endif









                    } break;
            }

            // might want to swap these out for shading programs...
            //// switch (mat->type)
            //// {
            ////     case MAT_WUV:
            ////         {
            ////             float w = (1.f - u - v);
            ////             v3 wuv = V3 (w, u, v);

            ////             res = wuv;

            ////         } break;

            ////     case MAT_NORMALS:
            ////         {



            ////             res = 0.5*V3 (N.x+1, N.y+1, N.z+1);

            ////         } break;

            ////     case MAT_LAMBERTIAN:
            ////         {
            ////             v3 rand = RandomInUnitSphere ();

            ////             v3 target = p + N + rand;

            ////             ray scattered = Ray (p, target - p);

            ////             Assert (mat->tex);
            ////             v3 albedo = GetAttenuation (mat->tex, 0, 0, p);

#if 1
            ////             // NOTE: (Kapsy) Facing ratio "lighting".
            ////             //// float facingratio = Clamp01 (Dot (-Unit (r->dir), N)*1.2f);
            ////             //// attenuation = attenuation*facingratio;//*Color (&scattered, depth+1);

            ////             // NOTE: (Kapsy) Directional Lighting.
            ////             //// v3 L = -g_light.dir;
            ////             //// v3 attenuation = albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L));

            ////             {
            ////             v3 L1 = -g_light.dir;
            ////             res = 0.6*albedo + 0.4*albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L1));

            ////             // NOTE: (Kapsy) Directional shadows.
            ////             float bias = 1e-4;

            ////             v3 L = -g_light.dir;

            ////             ray sray = {};
            ////             sray.orig = p + N*bias;
            ////             sray.dir = L;

            ////             hitrec shit = {};
            ////             shit.dist = MAXFLOAT;
            ////             TraverseTris (&sray, &object->aabb, &shit);

            ////             if (shit.dist < MAXFLOAT)
            ////             {
            ////                 res= res*0.3f;
            ////             }
            ////             else
            ////             {
            ////                 res= res;
            ////             }


            ////             }





            ////             v3 diff = {};
            ////             v3 spec = {};

            ////             pointlight_t *light = g_pointlights + 0;

            ////             float bias = 1e-4;
            ////             v3 lightdir = light->p - p;
            ////             float r2 = Dot (lightdir, lightdir);
            ////             float dist = sqrtf(r2);
            ////             lightdir/=dist;
            ////             v3 intensity = light->intensity*light->col/(4.f*M_PI*r2);

            ////             ray sray = {};
            ////             sray.orig = p + N*bias;
            ////             sray.dir = lightdir;

            ////             hitrec shit = {};
            ////             shit.dist = dist;
            ////             TraverseTris (&sray, &object->aabb, &shit);

            ////             float vis = 1.f;
            ////             if (shit.dist < dist)
            ////             {
            ////                 vis = 0.2f;
            ////             }

            ////             diff += vis*albedo*intensity*Clamp01 (Dot (N, lightdir));

            ////             v3 I = lightdir;
            ////             v3 R = I - 2*Dot (I, N)*N;

            ////             float n = 1500;
            ////             float Kd = 0.3;
            ////             float Ks = 0.02;

            ////             spec += vis*intensity*pow (Clamp01 (Dot (R, r->dir)), mat->Ns);

            ////             res += diff*mat->Kd + spec*mat->Ks;
#endif

#if 0
            ////             // NOTE: (Kapsy) Facing ratio "lighting".
            ////             //// float facingratio = Clamp01 (Dot (-Unit (r->dir), N)*1.2f);
            ////             //// attenuation = attenuation*facingratio;//*Color (&scattered, depth+1);

            ////             // NOTE: (Kapsy) Directional Lighting.
            ////             //// v3 L = -g_light.dir;
            ////             //// v3 attenuation = albedo/M_PI*g_light.intensity*g_light.col*Clamp01 (Dot (N, L));

            ////             // NOTE: (Kapsy) Directional shadows.
            ////             v3 attenuation = albedo;
            ////             float bias = 1e-4;

            ////             v3 L = -g_light.dir;

            ////             ray sray = {};
            ////             sray.orig = p + N*bias;
            ////             sray.dir = L;

            ////             hitrec shit = {};
            ////             shit.dist = MAXFLOAT;
            ////             TraverseTris (&sray, &object->aabb, &shit);

            ////             if (shit.dist < MAXFLOAT)
            ////             {
            ////                 attenuation = attenuation*0.3f;
            ////             }


            ////             // NOTE: (Kapsy) Point lights with shadows.

            ////             //// for (int i=0 ; i<g_pointlightscount ; i++)
            ////             //// {

            ////             ////     pointlight_t *light = g_pointlights + i;

            ////             ////     float bias = 1e-4;
            ////             ////     v3 lightdir = light->p - p;
            ////             ////     float r2 = Dot (lightdir, lightdir);
            ////             ////     float dist = sqrtf(r2);
            ////             ////     lightdir/=dist;
            ////             ////     v3 intensity = light->intensity*light->col/(4.f*M_PI*r2);

            ////             ////     ray sray = {};
            ////             ////     sray.orig = p + N*bias;
            ////             ////     sray.dir = lightdir;

            ////             ////     hitrec shit = {};
            ////             ////     shit.dist = dist;
            ////             ////     TraverseTris (&sray, &object->aabb, &shit);

            ////             ////     float viz = 1.f;
            ////             ////     if (shit.dist < dist)
            ////             ////     {
            ////             ////         viz = 0.2f;
            ////             ////     }

            ////             ////     res += viz*attenuation*intensity*Clamp01 (Dot (N, lightdir));
            ////             //// }


            ////             if (depth < MAX_DEPTH)
            ////                 res = attenuation;
            ////             else
            ////                 res = V3 (0);
#endif

            ////         } break;

            ////     case MAT_METAL:
            ////         {
            ////             v3 v = Unit (r->dir);
            ////             v3 reflected = v - 2*Dot (v, N)*N;

            ////             v3 bias = N*1e-4;

            ////             ray scattered = Ray (p + bias, reflected);// + mat->fuzz*RandomInUnitSphere ());

            ////             //// Assert (mat->tex);
            ////             //// v3 attenuation = GetAttenuation(mat->tex, 0, 0, p);

            ////             // checking if > 90 deg??
            ////             bool result = (Dot (scattered.dir, N) > 0.f);

            ////             if (depth < MAX_DEPTH && result)
            ////                 //res = attenuation*GetColorForRay (&scattered, object, depth+1);
            ////                 res = GetColorForRay (&scattered, object, depth+1);
            ////             else
            ////                 res = V3 (0.f);

            ////         } break;

            ////     case MAT_DIELECTRIC:
            ////         {
            ////             ray scattered;

            ////             v3 outwardnormal;
            ////             v3 reflected = Reflect (r->dir, N);
            ////             float niovernt;

            ////             float reflectprob;
            ////             float cos;


            ////             if (Dot (r->dir, N) > 0)
            ////             {
            ////                 outwardnormal = -N;
            ////                 niovernt = mat->refindex;
            ////                 cos = mat->refindex * Dot (r->dir, N) / Length (r->dir);
            ////             }
            ////             else
            ////             {
            ////                 outwardnormal = N;
            ////                 niovernt = 1.f/mat->refindex;
            ////                 cos = -Dot (r->dir, N) / Length (r->dir);
            ////             }

            ////             v3 refracted = V3 (0.f);

            ////             // Janky fix, good enough for now.
            ////             // Still not perfect, would like to try other (non Schlick) solutions
            ////             v3 bias = outwardnormal*1e-2;
            ////             p = p - bias;

            ////             v3 uv = Unit (r->dir);
            ////             float dt = Dot (uv, outwardnormal);
            ////             float discriminant = 1.f - niovernt*niovernt*(1-dt*dt);
            ////             if (discriminant > 0.f)
            ////             {
            ////                 refracted = niovernt*(uv - outwardnormal*dt) - outwardnormal*sqrt(discriminant);
            ////                 reflectprob = Schlick (cos, mat->refindex);
            ////             }
            ////             else
            ////             {
            ////                 scattered = Ray (p, reflected);
            ////                 reflectprob = 1.f;
            ////             }

            ////             if (drand48() < reflectprob)
            ////             {
            ////                 scattered = Ray (p, reflected);
            ////             }
            ////             else
            ////             {
            ////                 scattered = Ray (p, refracted);
            ////             }

            ////             if (depth < MAX_DEPTH)
            ////                 res = GetColorForRay (&scattered, object, depth+1);
            ////             else
            ////                 res = V3 (0.f);

            ////         } break;

            ////     default:
            ////         break;

            //// }
        }
        else
        {
            // NOTE: (Kapsy) Draw our psuedo sky background.
            v3 rdir = V3 (r->dir.x, r->dir.y, r->dir.z);

            float t = (Unit (rdir).y + 1.f)*0.5f;
            v3 cola = V3 (1.f);
            v3 colb = (1.f/255.f)*V3 (255.f, 128.f, 0.f);

            res = (1.f - t)*cola + t*colb;

            res = V3 (1,0,0);
        }

        res4.x[e] = res.x;
        res4.y[e] = res.y;
        res4.z[e] = res.z;
    }


    m128 zero = _mm_set1_ps (0.f);
    m128 all = _mm_set1_epi32 (0xffffffff);
    res4.x = zero + _mm_and_ps (res4.x, _mm_xor_ps (outmask, all));
    res4.y = zero + _mm_and_ps (res4.y, _mm_xor_ps (outmask, all));
    res4.z = zero + _mm_and_ps (res4.z, _mm_xor_ps (outmask, all));

#endif
