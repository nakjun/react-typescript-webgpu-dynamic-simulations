export class tritriIntersectionShader
{
    shader = `
    fn orient_2D(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
        return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
    }
    
    fn intersectionTestVertex(P1: vec2<f32>, Q1: vec2<f32>, R1: vec2<f32>, P2: vec2<f32>, Q2: vec2<f32>, R2: vec2<f32>) -> bool {
        if (orient_2D(R2, P2, Q1) >= 0.0) {
            if (orient_2D(R2, Q2, Q1) <= 0.0) {
                if (orient_2D(P1, P2, Q1) > 0.0) {
                    if (orient_2D(P1, Q2, Q1) <= 0.0) {
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    if (orient_2D(P1, P2, R1) >= 0.0) {
                        if (orient_2D(Q1, R1, P2) >= 0.0) {
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            } else {
                if (orient_2D(P1, Q2, Q1) <= 0.0) {
                    if (orient_2D(R2, Q2, R1) <= 0.0) {
                        if (orient_2D(Q1, R1, Q2) >= 0.0) {
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        } else {
            if (orient_2D(R2, P2, R1) >= 0.0) {
                if (orient_2D(Q1, R1, R2) >= 0.0) {
                    if (orient_2D(P1, P2, R1) >= 0.0) {
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    if (orient_2D(Q1, R1, Q2) >= 0.0) {
                        if (orient_2D(R2, R1, Q2) >= 0.0) {
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            } else {
                return false;
            }
        }
    }

    fn Intersection_test_edge(P1:vec2<f32>, Q1:vec2<f32>,R1:vec2<f32>, P2:vec2<f32>, Q2:vec2<f32>, R2:vec2<f32>) -> bool { 
        if (orient_2D(R2,P2,Q1) >= 0.0f) {
           if (orient_2D(P1,P2,Q1) >= 0.0f) { 
              if (orient_2D(P1,Q1,R2) >= 0.0f) {return true; }
              else {return false;}} else { 
                 if (orient_2D(Q1,R1,P2) >= 0.0f){ 
                    if (orient_2D(R1,P1,P2) >= 0.0f){return true;}
                    else {return false;}
                 } 
                 else {return false;} } 
        } else {
           if (orient_2D(R2,P2,R1) >= 0.0f) 
           {
              if (orient_2D(P1,P2,R1) >= 0.0f) 
              {
                 if (orient_2D(P1,R1,R2) >= 0.0f) 
                    {return true; }
                 else 
                 {
                    if (orient_2D(Q1,R1,R2) >= 0.0f)
                       {return true;}
                    else 
                       {return false;}
                 }
              }
              else  {return false;}
           }
           else 
              {return false; }
        }
     }

    fn ccw_tri_tri_intersection_2d(p1:vec2<f32>, q1:vec2<f32>, r1:vec2<f32>, p2:vec2<f32>, q2:vec2<f32>, r2:vec2<f32>) -> bool {
        if ( orient_2D(p2,q2,p1) >= 0.0f )
        {
            if ( orient_2D(q2,r2,p1) >= 0.0f ) 
            {
                if ( orient_2D(r2,p2,p1) >= 0.0f ) 
                {
                    { return true; }
                }
                else 
                {
                    return Intersection_test_edge(p1,q1,r1,p2,q2,r2);
                }
            } 
            else {  
                if ( orient_2D(r2,p2,p1) >= 0.0f ) 
                {
                    return Intersection_test_edge(p1,q1,r1,r2,p2,q2);
                }
                else 
                {
                    return intersectionTestVertex(p1,q1,r1,p2,q2,r2);
                }
            }
        }
        else {
            if ( orient_2D(q2,r2,p1) >= 0.0f ) {
                if ( orient_2D(r2,p2,p1) >= 0.0f ) 
                {
                    return Intersection_test_edge(p1,q1,r1,q2,r2,p2);
                }
                else  
                {
                    return intersectionTestVertex(p1,q1,r1,q2,r2,p2);
                }
            }
            else 
            {
                return intersectionTestVertex(p1,q1,r1,r2,p2,q2);
            }
        }
        }; 

    fn tri_tri_overlap_test_2d(p1:vec2<f32>, q1:vec2<f32>, r1:vec2<f32>, p2:vec2<f32>, q2:vec2<f32>, r2:vec2<f32>) -> bool {
        if ( orient_2D(p1,q1,r1) < 0.0f )
        {
            if ( orient_2D(p2,q2,r2) < 0.0f )
            {
                return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,r2,q2);
            }
            else
            {
                return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,q2,r2);
            }
        }
        else
        {
            if ( orient_2D(p2,q2,r2) < 0.0f )
            {
                return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,r2,q2);
            }
            else
            {
                return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,q2,r2);
            }
        }
    }

    fn coplanar_tri_tri3d(tri1: array<vec3<f32>, 3>, tri2: array<vec3<f32>, 3>, normal1: vec3<f32>, normal2: vec3<f32>) -> bool {
        var P1: vec2<f32>;
        var Q1: vec2<f32>;
        var R1: vec2<f32>;
        var P2: vec2<f32>;
        var Q2: vec2<f32>;
        var R2: vec2<f32>;
    
        let n_x = abs(normal1.x);
        let n_y = abs(normal1.y);
        let n_z = abs(normal1.z);
    
        // Projection of the triangles in 3D onto 2D such that the area of the projection is maximized.
        if (n_x > n_z && n_x >= n_y) {
            // Project onto plane YZ
            P1 = vec2<f32>(tri1[1].z, tri1[1].y);
            Q1 = vec2<f32>(tri1[0].z, tri1[0].y);
            R1 = vec2<f32>(tri1[2].z, tri1[2].y);
    
            P2 = vec2<f32>(tri2[1].z, tri2[1].y);
            Q2 = vec2<f32>(tri2[0].z, tri2[0].y);
            R2 = vec2<f32>(tri2[2].z, tri2[2].y);
        } else if (n_y > n_z && n_y >= n_x) {
            // Project onto plane XZ
            P1 = vec2<f32>(tri1[1].x, tri1[1].z);
            Q1 = vec2<f32>(tri1[0].x, tri1[0].z);
            R1 = vec2<f32>(tri1[2].x, tri1[2].z);
    
            P2 = vec2<f32>(tri2[1].x, tri2[1].z);
            Q2 = vec2<f32>(tri2[0].x, tri2[0].z);
            R2 = vec2<f32>(tri2[2].x, tri2[2].z);
        } else {
            // Project onto plane XY
            P1 = vec2<f32>(tri1[0].x, tri1[0].y);
            Q1 = vec2<f32>(tri1[1].x, tri1[1].y);
            R1 = vec2<f32>(tri1[2].x, tri1[2].y);
    
            P2 = vec2<f32>(tri2[0].x, tri2[0].y);
            Q2 = vec2<f32>(tri2[1].x, tri2[1].y);
            R2 = vec2<f32>(tri2[2].x, tri2[2].y);
        }
    
        return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);
    }
    
    fn check_min_max(p1: vec3<f32>, q1: vec3<f32>, r1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>, r2: vec3<f32>) -> bool {
        var v1: vec3<f32>;
        var v2: vec3<f32>;
        var n1: vec3<f32>;
    
        v1 = p2 - q1;
        v2 = p1 - q1;
        n1 = cross(v1, v2);
    
        v1 = q2 - q1;
        if (dot(v1, n1) > 0.0) {
            return false;
        }
    
        v1 = p2 - p1;
        v2 = r1 - p1;
        n1 = cross(v1, v2);
        v1 = r2 - p1;
        if (dot(v1, n1) > 0.0) {
            return false;
        } else {
            return true;
        }
    }

    fn tri_tri_3D(tri1: array<vec3<f32>, 3>, tri2: array<vec3<f32>, 3>, dist_tri2: vec3<f32>, n1: vec3<f32>, n2: vec3<f32>) -> bool {
        if (dist_tri2.x > 0.0) {
            if (dist_tri2.y > 0.0) {
                return check_min_max(tri1[0], tri1[2], tri1[1], tri2[2], tri2[0], tri2[1]);
            } else if (dist_tri2.z > 0.0) {
                return check_min_max(tri1[0], tri1[2], tri1[1], tri2[1], tri2[2], tri2[0]);
            } else {
                return check_min_max(tri1[0], tri1[1], tri1[2], tri2[0], tri2[1], tri2[2]);
            }
        } else if (dist_tri2.x < 0.0) {
            if (dist_tri2.y < 0.0) {
                return check_min_max(tri1[0], tri1[1], tri1[2], tri2[2], tri2[0], tri2[1]);
            } else if (dist_tri2.z < 0.0) {
                return check_min_max(tri1[0], tri1[1], tri1[2], tri2[1], tri2[2], tri2[0]);
            } else {
                return check_min_max(tri1[0], tri1[2], tri1[1], tri2[0], tri2[1], tri2[2]);
            }
        } else {
            if (dist_tri2.y < 0.0) {
                if (dist_tri2.z >= 0.0) {
                    return check_min_max(tri1[0], tri1[2], tri1[1], tri2[1], tri2[2], tri2[0]);
                } else {
                    return check_min_max(tri1[0], tri1[1], tri1[2], tri2[0], tri2[1], tri2[2]);
                }
            } else if (dist_tri2.y > 0.0) {
                if (dist_tri2.z > 0.0) {
                    return check_min_max(tri1[0], tri1[2], tri1[1], tri2[0], tri2[1], tri2[2]);
                } else {
                    return check_min_max(tri1[0], tri1[1], tri1[2], tri2[1], tri2[2], tri2[0]);
                }
            } else {
                if (dist_tri2.z > 0.0) {
                    return check_min_max(tri1[0], tri1[1], tri1[2], tri2[2], tri2[0], tri2[1]);
                } else if (dist_tri2.z < 0.0) {
                    return check_min_max(tri1[0], tri1[2], tri1[1], tri2[2], tri2[0], tri2[1]);
                } else {
                    return coplanar_tri_tri3d(tri1, tri2, n1, n2);
                }
            }
        }
    }
    
    fn tri_tri_overlap_3D(tri1: vec3<u32>, tri2: vec3<u32>, tri1_vtx: array<vec3<f32>, 3>, tri2_vtx: array<vec3<f32>, 3>) -> bool {
        // Placeholder for array size - ensure it matches your actual use case
        
        var n1: vec3<f32> = cross(tri1_vtx[1] - tri1_vtx[0], tri1_vtx[2] - tri1_vtx[0]);
        var n2: vec3<f32> = cross(tri2_vtx[1] - tri2_vtx[0], tri2_vtx[2] - tri2_vtx[0]);
    
        var dist_tri1: vec3<f32> = vec3<f32>(
            dot(n2, tri1_vtx[0] - tri2_vtx[0]),
            dot(n2, tri1_vtx[1] - tri2_vtx[0]),
            dot(n2, tri1_vtx[2] - tri2_vtx[0])
        );
        var dist_tri2: vec3<f32> = vec3<f32>(
            dot(n1, tri2_vtx[0] - tri1_vtx[0]),
            dot(n1, tri2_vtx[1] - tri1_vtx[0]),
            dot(n1, tri2_vtx[2] - tri1_vtx[0])
        );
    
        // Check for quick rejection
        if ((dist_tri1.x > 0.0 && dist_tri1.y > 0.0 && dist_tri1.z > 0.0) || (dist_tri1.x < 0.0 && dist_tri1.y < 0.0 && dist_tri1.z < 0.0)) {
            return false;
        }
        if ((dist_tri2.x > 0.0 && dist_tri2.y > 0.0 && dist_tri2.z > 0.0) || (dist_tri2.x < 0.0 && dist_tri2.y < 0.0 && dist_tri2.z < 0.0)) {
            return false;
        }

        var tmp1: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
            vec3(0.0,0.0,0.0), vec3(0.0,0.0,0.0), vec3(0.0,0.0,0.0)
        );
        var tmp2: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
            vec3(0.0,0.0,0.0), vec3(0.0,0.0,0.0), vec3(0.0,0.0,0.0)
        );
        var tmp3: vec3<f32> = vec3(0.0,0.0,0.0);

        if(dist_tri1.x > 0.0){
            if(dist_tri1.y > 0.0){
                tmp1[0]=tri1_vtx[2];
                tmp1[1]=tri1_vtx[0] ;
                tmp1[2]=tri1_vtx[1];

                tmp2[0]=tri2_vtx[0];
                tmp2[1]=tri2_vtx[2];
                tmp2[2]=tri2_vtx[1];

                tmp3.x=dist_tri2.x;
                tmp3.y=dist_tri2.z;
                tmp3.z=dist_tri2.y;

                return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
            }
            else if (dist_tri1.z > 0.0f) 
            {
                tmp1[0]=tri1_vtx[1];
                tmp1[1]=tri1_vtx[2] ;
                tmp1[2]=tri1_vtx[0];

                tmp2[0]=tri2_vtx[0];
                tmp2[1]=tri2_vtx[2];
                tmp2[2]=tri2_vtx[1];

                tmp3.x=dist_tri2.x;
                tmp3.y=dist_tri2.z;
                tmp3.z=dist_tri2.y;

                return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
            }
            else
            {
                tmp1[0]=tri1_vtx[0];
                tmp1[1]=tri1_vtx[1] ;
                tmp1[2]=tri1_vtx[2];

                tmp2[0]=tri2_vtx[0];
                tmp2[1]=tri2_vtx[1];
                tmp2[2]=tri2_vtx[2];

                tmp3.x=dist_tri2.x;
                tmp3.y=dist_tri2.y;
                tmp3.z=dist_tri2.z;
                return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
            }
        }
        else if (dist_tri1.x < 0.0f) 
        {
            if (dist_tri1.y < 0.0f) 
            {
               tmp1[0]=tri1_vtx[2];
               tmp1[1]=tri1_vtx[0] ;
               tmp1[2]=tri1_vtx[1];
      
               tmp2[0]=tri2_vtx[0];
               tmp2[1]=tri2_vtx[1];
               tmp2[2]=tri2_vtx[2];
      
               tmp3.x=dist_tri2.x;
               tmp3.y=dist_tri2.y;
               tmp3.z=dist_tri2.z;
               return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
            }
            else if (dist_tri1.z < 0.0f) 
            {
      
               tmp1[0]=tri1_vtx[1];
               tmp1[1]=tri1_vtx[2] ;
               tmp1[2]=tri1_vtx[0];
      
               tmp2[0]=tri2_vtx[0];
               tmp2[1]=tri2_vtx[1];
               tmp2[2]=tri2_vtx[2];
      
               tmp3.x=dist_tri2.x;
               tmp3.y=dist_tri2.y;
               tmp3.z=dist_tri2.z;
               return    tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
            }
            else
            {
               tmp1[0]=tri1_vtx[0];
               tmp1[1]=tri1_vtx[1] ;
               tmp1[2]=tri1_vtx[2];
      
               tmp2[0]=tri2_vtx[0];
               tmp2[1]=tri2_vtx[2];
               tmp2[2]=tri2_vtx[1];
      
               tmp3.x=dist_tri2.x;
               tmp3.y=dist_tri2.z;
               tmp3.z=dist_tri2.y;
               return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
            }
         } 
         else 
         {
            if (dist_tri1.y < 0.0f) {
               if (dist_tri1.z >= 0.0f) 
               {
                  tmp1[0]=tri1_vtx[1];
                  tmp1[1]=tri1_vtx[2] ;
                  tmp1[2]=tri1_vtx[0];
      
                  tmp2[0]=tri2_vtx[0];
                  tmp2[1]=tri2_vtx[2];
                  tmp2[2]=tri2_vtx[1];
      
                  tmp3.x=dist_tri2.x;
                  tmp3.y=dist_tri2.z;
                  tmp3.z=dist_tri2.y;
                  return    tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
               }
               else 
               {
                  tmp1[0]=tri1_vtx[0];
                  tmp1[1]=tri1_vtx[1] ;
                  tmp1[2]=tri1_vtx[2];
      
                  tmp2[0]=tri2_vtx[0];
                  tmp2[1]=tri2_vtx[1];
                  tmp2[2]=tri2_vtx[2];
      
                  tmp3.x=dist_tri2.x;
                  tmp3.y=dist_tri2.y;
                  tmp3.z=dist_tri2.z;
                  return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
               }
            }
            else if (dist_tri1.y > 0.0f) {
               if (dist_tri1.z > 0.0f) 
               {
                  tmp1[0]=tri1_vtx[0];
                  tmp1[1]=tri1_vtx[1] ;
                  tmp1[2]=tri1_vtx[2];
      
                  tmp2[0]=tri2_vtx[0];
                  tmp2[1]=tri2_vtx[2];
                  tmp2[2]=tri2_vtx[1];
      
                  tmp3.x=dist_tri2.x;
                  tmp3.y=dist_tri2.z;
                  tmp3.z=dist_tri2.y;
                  return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
               }
               else 
               {
                  tmp1[0]=tri1_vtx[1];
                  tmp1[1]=tri1_vtx[2] ;
                  tmp1[2]=tri1_vtx[0];
      
                  tmp2[0]=tri2_vtx[0];
                  tmp2[1]=tri2_vtx[1];
                  tmp2[2]=tri2_vtx[2];
      
                  tmp3.x=dist_tri2.x;
                  tmp3.y=dist_tri2.y;
                  tmp3.z=dist_tri2.z;
                  return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
               }
            }
            else  {
               if (dist_tri1.z > 0.0f) 
               {
                  tmp1[0]=tri1_vtx[2];
                  tmp1[1]=tri1_vtx[0] ;
                  tmp1[2]=tri1_vtx[1];
      
                  tmp2[0]=tri2_vtx[0];
                  tmp2[1]=tri2_vtx[1];
                  tmp2[2]=tri2_vtx[2];
      
                  tmp3.x=dist_tri2.x;
                  tmp3.y=dist_tri2.y;
                  tmp3.z=dist_tri2.z;
                  return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
      
               }
               else if (dist_tri1.z < 0.0f) 
               {
                  tmp1[0]=tri1_vtx[2];
                  tmp1[1]=tri1_vtx[0] ;
                  tmp1[2]=tri1_vtx[1];
      
                  tmp2[0]=tri2_vtx[0];
                  tmp2[1]=tri2_vtx[2];
                  tmp2[2]=tri2_vtx[1];
      
                  tmp3.x=dist_tri2.x;
                  tmp3.y=dist_tri2.z;
                  tmp3.z=dist_tri2.y;
                  return tri_tri_3D(tmp1,tmp2,tmp3,n1,n2);
               }
               else
               {
                  return coplanar_tri_tri3d(tri1_vtx,tri2_vtx,n1,n2);
               }
            }            
        }
    `;
}