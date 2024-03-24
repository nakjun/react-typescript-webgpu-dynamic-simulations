export class IntersectionShader {
    vertex_triangle_intersection_shader = `
    @group(0) @binding(0) var<storage, read_write> positionsCloth: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
    
    @group(0) @binding(2) var<storage, read_write> positionsObject: array<f32>;
    @group(0) @binding(3) var<storage, read_write> triangleObject: array<triangles_object>;
    
    @group(0) @binding(4) var<uniform> numParticles: u32;
    @group(0) @binding(5) var<uniform> numTrianglesObject: u32;

    @group(0) @binding(6) var<storage, read_write> tempBuffer: array<atomicI32>;
    @group(0) @binding(7) var<storage, read_write> fixed: array<u32>;    

    @group(0) @binding(8) var<storage, read_write> tempCountBuffer: array<atomicI32>;
    @group(0) @binding(9) var<storage, read_write> prevPosition: array<f32>;   

    struct atomicI32{
        value: atomic<i32>
    }

    struct triangles_object{
        v1:u32,
        v2:u32,
        v3:u32
    }

    fn getClothVertexPosition(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(positionsCloth[i], positionsCloth[i + 1u], positionsCloth[i + 2u]);
    }

    fn getPrevPosition(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(prevPosition[i], prevPosition[i + 1u], prevPosition[i + 2u]);
    }

    fn getClothVertexVelocity(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(velocities[i], velocities[i + 1u], velocities[i + 2u]);
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
    
        let epsilon = 0.01;
        if (a > -epsilon && a < epsilon) {
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
        if (t > epsilon && t < length(edgeVector)) { // Check if intersection is within the edge bounds
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

        //var targetIndex = x * numTrianglesObject + y;

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

        var threshold = 0.01;
        
        var rC = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], prev_pos, threshold);        
        var bC = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], prev_pos);

        var rC1 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], pos, threshold);        
        var bC1 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], pos);

        var rC2 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], next_pos, threshold);
        var bC2 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], next_pos);

        var targetValue = pos - (vel * deltaTime * 2) * 100.0;

        if( (rC && pointInTriangle(bC)) || (rC1 && pointInTriangle(bC1)) || (rC2 && pointInTriangle(bC2)) ){
            atomicAdd(&tempBuffer[x * 3 + 0].value, i32(targetValue.x));
            atomicAdd(&tempBuffer[x * 3 + 1].value, i32(targetValue.y));
            atomicAdd(&tempBuffer[x * 3 + 2].value, i32(targetValue.z));    
        }
    }

    @compute @workgroup_size(256)
    fn response(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x: u32 = global_id.x;
        
        if(x > numParticles) {return;}

        var fix = fixed[x];
        if(fix==1) {return;}

        var pos = getClothVertexPosition(x);        
        var vel = getClothVertexVelocity(x);

        let tempX = atomicLoad(&tempBuffer[x * 3 + 0].value);
        let tempY = atomicLoad(&tempBuffer[x * 3 + 1].value);
        let tempZ = atomicLoad(&tempBuffer[x * 3 + 2].value);
        let countBufferData = atomicLoad(&tempCountBuffer[x].value);

        var separatePos:vec3<f32> = vec3<f32>((f32(tempX) / 100.0) / f32(countBufferData), (f32(tempY) / 100.0) / f32(countBufferData), (f32(tempZ) / 100.0) / f32(countBufferData));        

        if(countBufferData>0)
        {
            vel *= -0.0;
            
            pos.x += (separatePos.x * 0.03);
            pos.y += (separatePos.y * 0.03);
            pos.z += (separatePos.z * 0.03);

            velocities[x*3 + 0] = vel.x;
            velocities[x*3 + 1] = vel.y;
            velocities[x*3 + 2] = vel.z;

            positionsCloth[x*3 + 0] = pos.x;
            positionsCloth[x*3 + 1] = pos.y;
            positionsCloth[x*3 + 2] = pos.z;
        }
        // if(countBufferData>0)
        // {
        //     vel *= -0.5;
            
        //     velocities[x*3 + 0] = vel.x;
        //     velocities[x*3 + 1] = vel.y;
        //     velocities[x*3 + 2] = vel.z;

        //     positionsCloth[x*3 + 0] = pos.x;
        //     positionsCloth[x*3 + 1] = pos.y;
        //     positionsCloth[x*3 + 2] = pos.z;
        // }      
    
        atomicStore(&tempCountBuffer[x].value, i32(0));
        atomicStore(&tempBuffer[x*3+0].value, i32(0));
        atomicStore(&tempBuffer[x*3+1].value, i32(0));
        atomicStore(&tempBuffer[x*3+2].value, i32(0));
    }
    `;

    getIntersectionShader() {
        return this.vertex_triangle_intersection_shader;
    }

    getTriTriIntersectionShader() {
        return this.tritriIntersectionShader;
    }

    tritriIntersectionShader = `
    
    @group(0) @binding(0) var<storage, read_write> positionsCloth: array<f32>;
    @group(0) @binding(1) var<storage, read_write> triangleCloth: array<triangles>;
    
    @group(0) @binding(2) var<storage, read_write> positionsObject: array<f32>;
    @group(0) @binding(3) var<storage, read_write> triangleObject: array<triangles>;
    
    @group(0) @binding(4) var<uniform> numTriangleCloth: u32;
    @group(0) @binding(5) var<uniform> numTrianglesObject: u32;

    @group(0) @binding(6) var<storage, read_write> tempBuffer: array<atomicI32>;
    @group(0) @binding(7) var<storage, read_write> tempCountBuffer: array<atomicI32>;

    @group(0) @binding(8) var<storage, read_write> velocityCloth: array<f32>;

    struct atomicI32{
        value: atomic<i32>
    }

    struct triangles{
        v1: u32,
        v2: u32,
        v3: u32,
    }

    fn getClothVertexPosition(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(positionsCloth[i], positionsCloth[i + 1u], positionsCloth[i + 2u]);
    }

    fn getClothVertexVelocity(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(velocityCloth[i], velocityCloth[i + 1u], velocityCloth[i + 2u]);
    }

    fn getObjectVertexPosition(index: u32) -> vec3<f32> {
        let i = index * 3u; // Assuming each vertex is represented by three consecutive floats (x, y, z)
        return vec3<f32>(positionsObject[i], positionsObject[i + 1u], positionsObject[i + 2u]);
    }

    fn calculateSpace(position: vec3<f32>) -> vec3<i32> {        
        return vec3<i32>(i32((position.x + 500.0) / 5.0), i32((position.y + 500.0) / 5.0), i32((position.z + 500.0) / 5.0));
    }

    fn distanceSquared(a: vec3<i32>, b: vec3<i32>) -> f32 {
        let delta = vec3<f32>(f32(a.x - b.x), f32(a.y - b.y), f32(a.z - b.z));
        return dot(delta, delta);
    }

    fn compareVec3(a: vec3<i32>, b: vec3<i32>) -> bool{
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    fn areTriangleSameVoxel(cloth_space1: vec3<i32>, cloth_space2: vec3<i32>, cloth_space3: vec3<i32>,
        object_space1: vec3<i32>, object_space2: vec3<i32>, object_space3: vec3<i32>) -> bool {

        if(compareVec3(cloth_space1, object_space1) || compareVec3(cloth_space1, object_space2) || compareVec3(cloth_space1, object_space3))
        {
            return true;
        }
        if(compareVec3(cloth_space2, object_space1) || compareVec3(cloth_space2, object_space2) || compareVec3(cloth_space2, object_space3))
        {
            return true;
        }
        if(compareVec3(cloth_space3, object_space1) || compareVec3(cloth_space3, object_space2) || compareVec3(cloth_space3, object_space3))
        {
            return true;
        }
        return false;
    }

    fn areTrianglesClose(cloth_space1: vec3<i32>, cloth_space2: vec3<i32>, cloth_space3: vec3<i32>,
        object_space1: vec3<i32>, object_space2: vec3<i32>, object_space3: vec3<i32>) -> bool {
        let minDistanceSquared = 100.0; // 충돌 검사를 할 최소 거리의 제곱값입니다. 10 단위로 10*10 = 100 입니다.

        // 각 삼각형의 공간 좌표의 중심점을 계산합니다.
        let cloth_center = (cloth_space1 + cloth_space2 + cloth_space3) / 3;
        let object_center = (object_space1 + object_space2 + object_space3) / 3;

        // 두 중심점 사이의 거리를 계산합니다.
        let distanceSquared = distanceSquared(cloth_center, object_center);

        // 두 삼각형의 중심점 사이의 거리가 충분히 가깝다면 true를 반환합니다.
        return distanceSquared < minDistanceSquared;
    }

    fn calculateNormal(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
        var u = p1 - p0;
        var v = p2 - p0;
        return normalize(cross(u, v));
    }
    
    fn direction(p0: vec3<f32>, p1: vec3<f32>) -> vec3<f32>
    {
        return normalize(p1 - p0);
    }

    fn isPointInsideTriangle(point: vec3<f32>, vertex0: vec3<f32>, vertex1: vec3<f32>, vertex2: vec3<f32>) -> bool {
        let normal = cross(vertex1 - vertex0, vertex2 - vertex0);
        let edge1 = vertex1 - vertex0;
        let vp1 = point - vertex0;
        if (dot(cross(edge1, vp1), normal) < 0.0) {
            return false;
        }
        let edge2 = vertex2 - vertex1;
        let vp2 = point - vertex1;
        if (dot(cross(edge2, vp2), normal) < 0.0) {
            return false;
        }
        let edge3 = vertex0 - vertex2;
        let vp3 = point - vertex2;
        if (dot(cross(edge3, vp3), normal) < 0.0) {
            return false;
        }
        return true;
    }

    struct CollisionResult {
        hit: bool,
        point: vec3<f32>,
    };

    fn intersect(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, src: vec3<f32>, dst: vec3<f32>) -> CollisionResult
    {
        let e1 = p1 - p0;
        let e2 = p2 - p0;

        let epsilon = 0.000001;

        var hit = vec3<f32>(0.0, 0.0, 0.0);

        var ray_direction = direction(src, dst);
        var ray_origin = src;
        
        // Calculate determinant
        let p: vec3<f32> = cross(ray_direction, e2);
    
        // Calculate determinant
        let det: f32 = dot(e1, p);
    
        // If determinant is near zero, ray lies in plane of triangle otherwise not
        if (det > -epsilon && det < epsilon) {
            // var c1 = isPointInsideTriangle(src, p0, p1, p2);
            // var c2 = isPointInsideTriangle(dst, p0, p1, p2);

            // if(c1) {hit = src;}
            // if(c2) {hit = dst;}

            // return CollisionResult(c1 || c2, hit);

            return CollisionResult(false, hit);
        }
        let invDet: f32 = 1.0 / det;
    
        // Calculate distance from p1 to ray origin
        let t: vec3<f32> = ray_origin - p1;
    
        // Calculate u parameter
        let u: f32 = dot(t, p) * invDet;
    
        // Check for ray hit
        if (u < 0.0 || u > 1.0) {
            return CollisionResult(false, hit);
        }
    
        // Prepare to test v parameter
        let q: vec3<f32> = cross(t, e1);
    
        // Calculate v parameter
        let v: f32 = dot(ray_direction, q) * invDet;
    
        // Check for ray hit
        if (v < 0.0 || u + v > 1.0) {
            return CollisionResult(false, hit);
        }
    
        // Intersection point
        hit = p1 + u * e1 + v * e2;
    
        if (dot(e2, q) * invDet > epsilon) {
            // Ray does intersect
            return CollisionResult(true, hit);
        }
    
        // No hit at all
        return CollisionResult(false, hit);
    }

    fn FindClosestVertex(p0:vec3<f32>,p1:vec3<f32>,p2:vec3<f32>,point:vec3<f32>) -> vec3<f32>{
        var minDistance: f32 = 1e10; // Simulating Mathf.Infinity
        var closestVertex: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        
        let distance0 = distance(p0, point);
        let distance1 = distance(p1, point);
        let distance2 = distance(p2, point);

        if (distance0 < minDistance) {
            minDistance = distance0;
            closestVertex = p0;
        }
        if (distance1 < minDistance) {
            minDistance = distance1;
            closestVertex = p1;
        }
        if (distance2 < minDistance) {
            minDistance = distance2;
            closestVertex = p2;
        }

        return closestVertex;
    }

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x: u32 = global_id.x;
        let y: u32 = global_id.y;
        
        if(x >= numTriangleCloth) {return;}
        if(y >= numTrianglesObject) {return;}

        var cloth_tri_info = triangleCloth[x];
        var object_tri_info = triangleObject[y];

        var f1: vec3<u32> = vec3<u32>(u32(cloth_tri_info.v1), u32(cloth_tri_info.v2), u32(cloth_tri_info.v3));
        var f2: vec3<u32> = vec3<u32>(u32(object_tri_info.v1), u32(object_tri_info.v2), u32(object_tri_info.v3));

        //공간 계산

        var tri1_vtx: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
            getClothVertexPosition(f1.x),
            getClothVertexPosition(f1.y),
            getClothVertexPosition(f1.z)
        );
        var tri2_vtx: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
            getObjectVertexPosition(f2.x),
            getObjectVertexPosition(f2.y),
            getObjectVertexPosition(f2.z)
        );            
        
        var cloth_space1 = calculateSpace(tri1_vtx[0]);
        var cloth_space2 = calculateSpace(tri1_vtx[1]);
        var cloth_space3 = calculateSpace(tri1_vtx[2]);

        var object_space1 = calculateSpace(tri2_vtx[0]);
        var object_space2 = calculateSpace(tri2_vtx[1]);
        var object_space3 = calculateSpace(tri2_vtx[2]);

        if (!areTriangleSameVoxel(cloth_space1, cloth_space2, cloth_space3,
            object_space1, object_space2, object_space3)) {
            // 충분히 가까운 공간에 있지 않다면, 충돌 검사를 수행하지 않습니다.
            return;
        }

        //var result = tri_tri_overlap_3D(f1, f2, tri1_vtx, tri2_vtx);

        var res1 = intersect(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[0], tri1_vtx[1]);
        var res2 = intersect(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[0], tri1_vtx[2]);
        var res3 = intersect(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[1], tri1_vtx[2]);
        var collisionPoint:vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        if(!res1.hit && !res2.hit && !res3.hit) { return; }
        
        var count:i32 = 0;
        if(res1.hit) {
            collisionPoint += res1.point;
            count += 1;
        }
        if(res2.hit) {
            collisionPoint += res2.point;
            count += 1;
        }
        if(res3.hit) {
            collisionPoint += res3.point;
            count += 1;
        }

        collisionPoint = collisionPoint / f32(count);
               
        var closestVertex = FindClosestVertex(tri1_vtx[0], tri1_vtx[1], tri1_vtx[2], collisionPoint);
        var triNormal = (tri1_vtx[0] + tri1_vtx[1] + tri1_vtx[2]) / 3.0;
        //var separationVector = normalize(closestVertex - collisionPoint) * 0.0001 + triNormal * 0.0001;
        var separationVector = triNormal * 0.07;
        if(closestVertex.x==tri1_vtx[0].x && closestVertex.y==tri1_vtx[0].y && closestVertex.z==tri1_vtx[0].z)
        {
            atomicAdd(&tempBuffer[f1.x * 3 + 0].value, i32(separationVector.x * 100.0));
            atomicAdd(&tempBuffer[f1.x * 3 + 1].value, i32(separationVector.y * 100.0));
            atomicAdd(&tempBuffer[f1.x * 3 + 2].value, i32(separationVector.z * 100.0));
            atomicAdd(&tempCountBuffer[f1.x].value,i32(1));        
        }
        else if(closestVertex.x==tri1_vtx[1].x && closestVertex.y==tri1_vtx[1].y && closestVertex.z==tri1_vtx[1].z)
        {
            atomicAdd(&tempBuffer[f1.y * 3 + 0].value, i32(separationVector.x * 100.0));
            atomicAdd(&tempBuffer[f1.y * 3 + 1].value, i32(separationVector.y * 100.0));
            atomicAdd(&tempBuffer[f1.y * 3 + 2].value, i32(separationVector.z * 100.0));
            atomicAdd(&tempCountBuffer[f1.y].value,i32(1));
        }
        else if(closestVertex.x==tri1_vtx[2].x && closestVertex.y==tri1_vtx[2].y && closestVertex.z==tri1_vtx[2].z)
        {            
            atomicAdd(&tempBuffer[f1.z * 3 + 0].value, i32(separationVector.x * 100.0));
            atomicAdd(&tempBuffer[f1.z * 3 + 1].value, i32(separationVector.y * 100.0));
            atomicAdd(&tempBuffer[f1.z * 3 + 2].value, i32(separationVector.z * 100.0));
            atomicAdd(&tempCountBuffer[f1.z].value,i32(1));
        }
        
        // if(res1.hit && res2.hit)
        // {
        //     var dir1 = normalize(tri1_vtx[0]-res1.point);
        //     var dir2 = normalize(tri1_vtx[0]-res2.point);

        //     var rPoints1 = res1.point + res1.point * dir1;
        //     var rPoints2 = res2.point + res2.point * dir2;

        //     var diff = (rPoints1 + rPoints2) / 2.0;

        //     atomicAdd(&tempBuffer[f1.x * 3 + 0].value, i32(diff.x * 100.0));
        //     atomicAdd(&tempBuffer[f1.x * 3 + 1].value, i32(diff.y * 100.0));
        //     atomicAdd(&tempBuffer[f1.x * 3 + 2].value, i32(diff.z * 100.0));
        //     atomicAdd(&tempCountBuffer[f1.x].value,i32(1));
        // }
        // if(res1.hit && res3.hit)
        // {
        //     var dir1 = normalize(tri1_vtx[1]-res1.point);
        //     var dir2 = normalize(tri1_vtx[1]-res3.point);

        //     var rPoints1 = res1.point + res1.point * dir1;
        //     var rPoints2 = res3.point + res3.point * dir2;

        //     var diff = (rPoints1 + rPoints2) / 2.0;

        //     atomicAdd(&tempBuffer[f1.y * 3 + 0].value, i32(diff.x * 100.0));
        //     atomicAdd(&tempBuffer[f1.y * 3 + 1].value, i32(diff.y * 100.0));
        //     atomicAdd(&tempBuffer[f1.y * 3 + 2].value, i32(diff.z * 100.0));
        //     atomicAdd(&tempCountBuffer[f1.y].value,i32(1));
        // }
        // if(res2.hit && res3.hit)
        // {
        //     var dir1 = normalize(tri1_vtx[2]-res2.point);
        //     var dir2 = normalize(tri1_vtx[2]-res3.point);

        //     var rPoints1 = res2.point + res2.point * dir1;
        //     var rPoints2 = res3.point + res3.point * dir2;

        //     var diff = (rPoints1 + rPoints2) / 2.0;

        //     atomicAdd(&tempBuffer[f1.z * 3 + 0].value, i32(diff.x * 100.0));
        //     atomicAdd(&tempBuffer[f1.z * 3 + 1].value, i32(diff.y * 100.0));
        //     atomicAdd(&tempBuffer[f1.z * 3 + 2].value, i32(diff.z * 100.0));
        //     atomicAdd(&tempCountBuffer[f1.z].value,i32(1));
        // }        
    }
    `;
}