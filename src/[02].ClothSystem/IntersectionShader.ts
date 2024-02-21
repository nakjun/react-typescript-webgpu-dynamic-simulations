export class IntersectionShader {
    intersection_shader = `
    @group(0) @binding(0) var<storage, read_write> positionsCloth: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
    
    @group(0) @binding(2) var<storage, read_write> positionsObject: array<f32>;
    @group(0) @binding(3) var<storage, read_write> triangleObject: array<triangles_object>;
    
    @group(0) @binding(4) var<uniform> numParticles: u32;
    @group(0) @binding(5) var<uniform> numTrianglesObject: u32;

    @group(0) @binding(6) var<storage, read_write> tempBuffer: array<f32>;

    @group(0) @binding(7) var<storage, read_write> fixed: array<u32>;

    struct triangles_cloth{
        v1: f32,
        v2: f32,
        v3: f32,
        t1: f32,
        t2: f32,
        t3: f32,
    }

    struct triangles_object{
        v1:u32,
        v2:u32,
        v3:u32
    }

    struct vec2f{
        x:f32,
        y:f32
    }
    
    fn getClothVertexPosition(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(positionsCloth[i], positionsCloth[i + 1u], positionsCloth[i + 2u]);
    }

    fn getClothVertexVelocity(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(velocities[i], velocities[i + 1u], velocities[i + 2u]);
    }

    fn getTempBuffer(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(tempBuffer[i], tempBuffer[i + 1u], tempBuffer[i + 2u]);
    }

    fn getObjectVertexPosition(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(positionsObject[i], positionsObject[i + 1u], positionsObject[i + 2u]);
    }
    
    fn barycentricCoords(A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, P: vec3<f32>) -> vec3<f32> {
        let v0 = B - A;
        let v1 = C - A;
        let v2 = P - A;

        let d00 = dot(v0, v0);
        let d01 = dot(v0, v1);
        let d11 = dot(v1, v1);
        let d20 = dot(v2, v0);
        let d21 = dot(v2, v1);
        let denom = d00 * d11 - d01 * d01;
    
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;
    
        return vec3<f32>(u, v, w);
    }
    
    fn pointInTriangle(barycentricCoords: vec3<f32>) -> bool {
        let u = barycentricCoords.x;
        let v = barycentricCoords.y;
        let w = barycentricCoords.z;
    
        return (u >= 0.0) && (v >= 0.0) && (w >= 0.0) && (u + v + w <= 1.0);
    }

    fn isPointInPlane(A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, P: vec3<f32>, epsilon: f32) -> bool {
        let planeNormal = cross(B - A, C - A);
        let pointVector = P - A;
        let dotProduct = dot(planeNormal, pointVector);

        return abs(dotProduct) <= epsilon;
    }

    fn isPointInPlane2(A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, P: vec3<f32>, epsilon: f32) -> f32 {
        let planeNormal = cross(B - A, C - A);
        let pointVector = P - A;
        let dotProduct = dot(planeNormal, pointVector);
        return abs(dotProduct);
    }
    
    struct CollisionResult {
        hit: bool,
        point: vec3<f32>,
    };
    
    fn checkEdgeTriangleCollision(startPos: vec3<f32>, endPos: vec3<f32>, triangleVertices: array<vec3<f32>, 3>) -> CollisionResult {
        let edgeVector = endPos - startPos;
        let rayVector = normalize(edgeVector);
        let edge1 = triangleVertices[1] - triangleVertices[0];
        let edge2 = triangleVertices[2] - triangleVertices[0];
        let h = cross(rayVector, edge2);
        let a = dot(edge1, h);
    
        let EPSILON = 0.01;
        if (a > -EPSILON && a < EPSILON) {
            return CollisionResult(false, vec3<f32>(0.0, 0.0, 0.0)); // Parallel, no intersection
        }
    
        let f = 1.0 / a;
        let s = startPos - triangleVertices[0];
        let u = f * dot(s, h);
        if (u < 0.0 || u > 1.0) {
            return CollisionResult(false, vec3<f32>(0.0, 0.0, 0.0));
        }
    
        let q = cross(s, edge1);
        let v = f * dot(rayVector, q);
        if (v < 0.0 || u + v > 1.0) {
            return CollisionResult(false, vec3<f32>(0.0, 0.0, 0.0));
        }
    
        let t = f * dot(edge2, q);
        if (t > EPSILON && t < length(edgeVector)) { // Check if intersection is within the edge bounds
            let intersectionPoint = startPos + t * rayVector; // Calculate intersection point
            return CollisionResult(true, intersectionPoint);
        }
        return CollisionResult(false, vec3<f32>(0.0, 0.0, 0.0));
    }
    
    fn calculateNormal(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
        var u = p1 - p0;
        var v = p2 - p0;
        return normalize(cross(u, v));
    }

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x: u32 = global_id.x;
        let y: u32 = global_id.y;
        
        if(x >= numParticles) {return;}
        if(y >= numTrianglesObject) {return;}

        var targetIndex = x * numTrianglesObject + y;

        var fix = fixed[x];
        if(fix==1) {return;}

        var pos = getClothVertexPosition(x);
        var vel = getClothVertexVelocity(x);

        var object_tri_info = triangleObject[y];
        var f2: vec3<u32> = vec3<u32>(u32(object_tri_info.v1), u32(object_tri_info.v2), u32(object_tri_info.v3));

        var tri2_vtx: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
            getObjectVertexPosition(f2.x),
            getObjectVertexPosition(f2.y),
            getObjectVertexPosition(f2.z)
        );

        var tri_normal = calculateNormal(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]);

        var deltaTime: f32 = 0.001; // Assuming 60 FPS for simplicity
        
        var next_pos = pos + (vel * deltaTime);
        var prev_pos = pos - (vel * deltaTime);

        var threshold = 0.2;

        var edgeTriangleCollisionResult1 = checkEdgeTriangleCollision(pos, next_pos, tri2_vtx);        
        var edgeTriangleCollisionResult2 = checkEdgeTriangleCollision(prev_pos, pos, tri2_vtx);        

        var rC = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], prev_pos, threshold);        
        var bC = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], prev_pos);

        var rC1 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], pos, threshold);        
        var bC1 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], pos);

        var rC2 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], next_pos, threshold);
        var bC2 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], next_pos);

        if( (rC && pointInTriangle(bC)) || (rC1 && pointInTriangle(bC1)) || (rC2 && pointInTriangle(bC2)) ){        
            tempBuffer[targetIndex*3 + 0] = tri_normal.x * 2.0;
            tempBuffer[targetIndex*3 + 1] = tri_normal.y * 2.0;
            tempBuffer[targetIndex*3 + 2] = tri_normal.z * 2.0;
        }
        else{
            tempBuffer[targetIndex*3 + 0] = 0.0;
            tempBuffer[targetIndex*3 + 1] = 0.0;
            tempBuffer[targetIndex*3 + 2] = 0.0;
        }
    }

    @compute @workgroup_size(256)
    fn summation(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x: u32 = global_id.x;
        
        if(x > numParticles) {return;}

        var fix = fixed[x];
        if(fix==1) {return;}

        var pos = getClothVertexPosition(x);
        var vel = getClothVertexVelocity(x);

        var start = x * numTrianglesObject;
        var end = (x+1) * numTrianglesObject;

        var newPos = vec3<f32>(0.0, 0.0, 0.0);
        var count = 0;

        for(var i: u32 = start; i < end; i++) {
            var data = getTempBuffer(i);                        
            newPos.x += data.x;
            newPos.y += data.y;
            newPos.z += data.z;
        }

        newPos.x /= f32(numTrianglesObject);
        newPos.y /= f32(numTrianglesObject);
        newPos.z /= f32(numTrianglesObject);

        pos.x += (newPos.x);
        pos.y += (newPos.y);
        pos.z += (newPos.z);

        var threshold:f32 = 0.000001;

        if(newPos.x<=threshold && newPos.y<=threshold && newPos.z<=threshold){
                
        }else{
            //fixed[x] = 1;
            
            vel *= 0.01;

            velocities[x*3 + 0] = vel.x;
            velocities[x*3 + 1] = vel.y;
            velocities[x*3 + 2] = vel.z;

        }
        positionsCloth[x*3 + 0] = pos.x;
        positionsCloth[x*3 + 1] = pos.y;
        positionsCloth[x*3 + 2] = pos.z;    
    }
    `;

    getIntersectionShader() {
        return this.intersection_shader;
    }

    tritriIntersectionShader = `
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
        
        // var tri1_vtx: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
        //     positions[tri1.x], positions[tri1.y], positions[tri1.z],
        // );
        // var tri2_vtx: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
        //     positions[tri2.x], positions[tri2.y], positions[tri2.z],
        // );
    
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
    }
    `;
}