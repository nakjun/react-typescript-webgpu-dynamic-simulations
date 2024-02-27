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
    fn summation(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
        
        var newPos = vec3<f32>(0.0, 0.0, 0.0);

        var update_value = vec3<f32>(
            f32(tempX / countBufferData) /1000.0, 
            f32(tempY / countBufferData) /1000.0, 
            f32(tempZ / countBufferData) /1000.0);

        newPos.x = pos.x + update_value.x;
        newPos.y = pos.y + update_value.y;
        newPos.z = pos.z + update_value.z;
    
        var threshold:f32 = 0.0000001;
        
        
        if(distance(pos,newPos) > threshold) {
            vel *= -0.0001;
            
            velocities[x*3 + 0] = vel.x;
            velocities[x*3 + 1] = vel.y;
            velocities[x*3 + 2] = vel.z;
            
            positionsCloth[x*3 + 0] = newPos.x;
            positionsCloth[x*3 + 1] = newPos.y;
            positionsCloth[x*3 + 2] = newPos.z;                
        }

        atomicStore(&tempBuffer[x * 3 + 0].value, i32(0));
        atomicStore(&tempBuffer[x * 3 + 1].value, i32(0));
        atomicStore(&tempBuffer[x * 3 + 2].value, i32(0));
    
        atomicStore(&tempCountBuffer[x].value, i32(0));
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
        return vec3<i32>(i32((position.x + 500.0) / 10.0), i32((position.y + 500.0) / 10.0), i32((position.z + 500.0) / 10.0));
    }

    fn distanceSquared(a: vec3<i32>, b: vec3<i32>) -> f32 {
        let delta = vec3<f32>(f32(a.x - b.x), f32(a.y - b.y), f32(a.z - b.z));
        return dot(delta, delta);
    }

    fn areTrianglesClose(cloth_space1: vec3<i32>, cloth_space2: vec3<i32>, cloth_space3: vec3<i32>,
        object_space1: vec3<i32>, object_space2: vec3<i32>, object_space3: vec3<i32>) -> bool {
        let minDistanceSquared = 5.0; // 충돌 검사를 할 최소 거리의 제곱값입니다. 10 단위로 10*10 = 100 입니다.

        // 각 삼각형의 공간 좌표의 중심점을 계산합니다.
        let cloth_center = (cloth_space1 + cloth_space2 + cloth_space3) / 3;
        let object_center = (object_space1 + object_space2 + object_space3) / 3;

        // 두 중심점 사이의 거리를 계산합니다.
        let distanceSquared = distanceSquared(cloth_center, object_center);

        // 두 삼각형의 중심점 사이의 거리가 충분히 가깝다면 true를 반환합니다.
        return distanceSquared < minDistanceSquared;
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

    fn dot2(v: vec3<f32>) -> f32 {
        return dot(v, v);
    }
    
    fn constrain(v: f32, minVal: f32, maxVal: f32) -> f32 {
        return max(minVal, min(maxVal, v));
    }
    
    fn sdfTriangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
        let ba = b - a; 
        let pa = p - a;
        let cb = c - b; 
        let pb = p - b;
        let ac = a - c; 
        let pc = p - c;
        let nor = cross(ba, ac);
    
        var insideTest = sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc));
        var value: f32;
    
        if (insideTest < 2.0) {
            value = min(
                min(
                    dot2((ba * constrain(dot(ba, pa) / dot2(ba), 0.0, 1.0)) - pa),
                    dot2((cb * constrain(dot(cb, pb) / dot2(cb), 0.0, 1.0)) - pb)
                ),
                dot2((ac * constrain(dot(ac, pc) / dot2(ac), 0.0, 1.0)) - pc)
            );
        } else {
            value = dot(nor, pa) * dot(nor, pa) / dot2(nor);
        }
    
        return sqrt(value);
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

        if (!areTrianglesClose(cloth_space1, cloth_space2, cloth_space3,
            object_space1, object_space2, object_space3)) {
            // 충분히 가까운 공간에 있지 않다면, 충돌 검사를 수행하지 않습니다.
            return;
        }

        var result = tri_tri_overlap_3D(f1, f2, tri1_vtx, tri2_vtx);        
        let threshold = 10.0;
        if(result){

            // var rC1 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[0], threshold);        
            // var bC1 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[0]);

            // var rC2 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[1], threshold);        
            // var bC2 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[1]);

            // var rC3 = isPointInPlane(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[2], threshold);        
            // var bC3 = barycentricCoords(tri2_vtx[0], tri2_vtx[1], tri2_vtx[2], tri1_vtx[2]);

            // if( (rC1 && pointInTriangle(bC1) ) ){

            //     var targetValue = (getClothVertexVelocity(f1.x) * 0.0001) * 100.0;

            //     atomicAdd(&tempBuffer[f1.x * 3 + 0].value, i32(targetValue.x));
            //     atomicAdd(&tempBuffer[f1.x * 3 + 1].value, i32(targetValue.y));
            //     atomicAdd(&tempBuffer[f1.x * 3 + 2].value, i32(targetValue.z));    
            //     atomicAdd(&tempCountBuffer[f1.x].value, i32(1));
            // }

            // if( (rC2 && pointInTriangle(bC2) ) ){

            //     var targetValue = (getClothVertexVelocity(f1.y) * 0.0001) * 100.0;

            //     atomicAdd(&tempBuffer[f1.y * 3 + 0].value, i32(targetValue.x));
            //     atomicAdd(&tempBuffer[f1.y * 3 + 1].value, i32(targetValue.y));
            //     atomicAdd(&tempBuffer[f1.y * 3 + 2].value, i32(targetValue.z));    
            //     atomicAdd(&tempCountBuffer[f1.y].value, i32(1));
            // }

            // if( (rC2 && pointInTriangle(bC2) ) ){

            //     var targetValue = (getClothVertexVelocity(f1.z) * 0.0001) * 100.0;

            //     atomicAdd(&tempBuffer[f1.z * 3 + 0].value, i32(targetValue.x));
            //     atomicAdd(&tempBuffer[f1.z * 3 + 1].value, i32(targetValue.y));
            //     atomicAdd(&tempBuffer[f1.z * 3 + 2].value, i32(targetValue.z));    
            //     atomicAdd(&tempCountBuffer[f1.z].value, i32(1));
            // }

            // ----------------------------------------------------------------

            // var res0 = checkEdgeTriangleCollision(tri1_vtx[2], tri1_vtx[0], tri2_vtx);
            // var res1 = checkEdgeTriangleCollision(tri1_vtx[2], tri1_vtx[1], tri2_vtx);
            // var res2 = checkEdgeTriangleCollision(tri1_vtx[1], tri1_vtx[0], tri2_vtx);

            // if(res0.hit){

            //     var targetValue1 = (tri1_vtx[0] - res0.point) * 100.0;
            //     var targetValue2 = (tri1_vtx[2] - res0.point) * 100.0;

            //     atomicAdd(&tempBuffer[f1.x * 3 + 0].value, i32(targetValue1.x));
            //     atomicAdd(&tempBuffer[f1.x * 3 + 1].value, i32(targetValue1.y));
            //     atomicAdd(&tempBuffer[f1.x * 3 + 2].value, i32(targetValue1.z));
    
            //     atomicAdd(&tempBuffer[f1.z * 3 + 0].value, i32(targetValue2.x));
            //     atomicAdd(&tempBuffer[f1.z * 3 + 1].value, i32(targetValue2.y));
            //     atomicAdd(&tempBuffer[f1.z * 3 + 2].value, i32(targetValue2.z));

            //     atomicAdd(&tempCountBuffer[f1.x].value, i32(1));
            //     atomicAdd(&tempCountBuffer[f1.z].value, i32(1));
            // }

            // if(res1.hit){

            //     var targetValue1 = (tri1_vtx[1] - res0.point) * 100.0;
            //     var targetValue2 = (tri1_vtx[2] - res0.point) * 100.0;

            //     atomicAdd(&tempBuffer[f1.y * 3 + 0].value, i32(targetValue1.x));
            //     atomicAdd(&tempBuffer[f1.y * 3 + 1].value, i32(targetValue1.y));
            //     atomicAdd(&tempBuffer[f1.y * 3 + 2].value, i32(targetValue1.z));
    
            //     atomicAdd(&tempBuffer[f1.z * 3 + 0].value, i32(targetValue2.x));
            //     atomicAdd(&tempBuffer[f1.z * 3 + 1].value, i32(targetValue2.y));
            //     atomicAdd(&tempBuffer[f1.z * 3 + 2].value, i32(targetValue2.z));

            //     atomicAdd(&tempCountBuffer[f1.y].value, i32(1));
            //     atomicAdd(&tempCountBuffer[f1.z].value, i32(1));
            // }

            // if(res2.hit){

            //     var targetValue1 = (tri1_vtx[0] - res0.point) * 100.0;
            //     var targetValue2 = (tri1_vtx[1] - res0.point) * 100.0;

            //     atomicAdd(&tempBuffer[f1.x * 3 + 0].value, i32(targetValue1.x));
            //     atomicAdd(&tempBuffer[f1.x * 3 + 1].value, i32(targetValue1.y));
            //     atomicAdd(&tempBuffer[f1.x * 3 + 2].value, i32(targetValue1.z));
    
            //     atomicAdd(&tempBuffer[f1.y * 3 + 0].value, i32(targetValue2.x));
            //     atomicAdd(&tempBuffer[f1.y * 3 + 1].value, i32(targetValue2.y));
            //     atomicAdd(&tempBuffer[f1.y * 3 + 2].value, i32(targetValue2.z));

            //     atomicAdd(&tempCountBuffer[f1.x].value, i32(1));
            //     atomicAdd(&tempCountBuffer[f1.y].value, i32(1));
            // }

            // ----------------------------------------------------------------

            // atomicStore(&tempCountBuffer[f1.x].value, i32(sdfTriangle(tri1_vtx[0], tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]) *100.0));
            // atomicStore(&tempCountBuffer[f1.y].value, i32(sdfTriangle(tri1_vtx[1], tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]) *100.0));
            // atomicStore(&tempCountBuffer[f1.z].value, i32(sdfTriangle(tri1_vtx[2], tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]) *100.0));

            if(sdfTriangle(tri1_vtx[0], tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]) < 500.0)
            {
                var targetValue = -(getClothVertexVelocity(f1.x) * 0.001) * 1000.0;
                atomicAdd(&tempBuffer[f1.x * 3 + 0].value, i32(targetValue.x));
                atomicAdd(&tempBuffer[f1.x * 3 + 1].value, i32(targetValue.y));
                atomicAdd(&tempBuffer[f1.x * 3 + 2].value, i32(targetValue.z));    
                atomicAdd(&tempCountBuffer[f1.x].value, i32(1));
            }

            if(sdfTriangle(tri1_vtx[1], tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]) < 500.0)
            {
                var targetValue = -(getClothVertexVelocity(f1.y) * 0.001) * 1000.0;
                atomicAdd(&tempBuffer[f1.y * 3 + 0].value, i32(targetValue.x));
                atomicAdd(&tempBuffer[f1.y * 3 + 1].value, i32(targetValue.y));
                atomicAdd(&tempBuffer[f1.y * 3 + 2].value, i32(targetValue.z));    
                atomicAdd(&tempCountBuffer[f1.y].value, i32(1));
            }

            if(sdfTriangle(tri1_vtx[2], tri2_vtx[0], tri2_vtx[1], tri2_vtx[2]) < 500.0)
            {
                var targetValue = -(getClothVertexVelocity(f1.z) * 0.001) * 1000.0;
                atomicAdd(&tempBuffer[f1.z * 3 + 0].value, i32(targetValue.x));
                atomicAdd(&tempBuffer[f1.z * 3 + 1].value, i32(targetValue.y));
                atomicAdd(&tempBuffer[f1.z * 3 + 2].value, i32(targetValue.z));    
                atomicAdd(&tempCountBuffer[f1.z].value, i32(1));
            }
        }
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
    }
    `;
}