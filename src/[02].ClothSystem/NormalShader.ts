export class NormalShader{
    normalUpdateComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> triangleCalculationBuffer: array<triangles>;
    @group(0) @binding(2) var<storage, read_write> tempNormal: array<atomicI32>;
    @group(0) @binding(3) var<uniform> numTriangles: u32;

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }

    fn calculateNormal(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
        var u = p1 - p0;
        var v = p2 - p0;
        return normalize(cross(u, v));
    }

    struct triangles{
        v1: f32,
        v2: f32,
        v3: f32,
    }

    struct atomicI32{
        value: atomic<i32>
    }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let id: u32 = global_id.x;
        if (id >= numTriangles) {
            return;
        }

        // Accessing triangle calculation data
        var triangle = triangleCalculationBuffer[id];
        var v1: u32 = u32(triangle.v1);
        var v2: u32 = u32(triangle.v2);
        var v3: u32 = u32(triangle.v3);

        // Fetching positions from the positions buffer
        var p0: vec3<f32> = getPosition(v1);
        var p1: vec3<f32> = getPosition(v2);
        var p2: vec3<f32> = getPosition(v3);

        // Calculate the normal for this triangle
        var normal: vec3<f32> = calculateNormal(p0, p1, p2);

        atomicAdd(&tempNormal[v1 * 3 + 0].value, i32(normal.x*100));
        atomicAdd(&tempNormal[v1 * 3 + 1].value, i32(normal.y*100));
        atomicAdd(&tempNormal[v1 * 3 + 2].value, i32(normal.z*100));

        atomicAdd(&tempNormal[v2 * 3 + 0].value, i32(normal.x*100));
        atomicAdd(&tempNormal[v2 * 3 + 1].value, i32(normal.y*100));
        atomicAdd(&tempNormal[v2 * 3 + 2].value, i32(normal.z*100));

        atomicAdd(&tempNormal[v3 * 3 + 0].value, i32(normal.x*100));
        atomicAdd(&tempNormal[v3 * 3 + 1].value, i32(normal.y*100));
        atomicAdd(&tempNormal[v3 * 3 + 2].value, i32(normal.z*100));
    }
    `;

    getNormalUpdateComputeShader(){
        return this.normalUpdateComputeShader;
    }

    normalSummationComputeShader = `
        @group(0) @binding(0) var<storage, read_write> tempNormal: array<atomicI32>;
        @group(0) @binding(1) var<storage, read_write> normal: array<f32>;
        @group(0) @binding(2) var<uniform> maxConnectedTriangle: u32;
        @group(0) @binding(3) var<uniform> numParticles: u32;

        fn getNormal(index:u32) -> vec3<f32>{
            return vec3<f32>(normal[index*3],normal[index*3+1],normal[index*3+2]);
        }

        struct atomicI32{
            value: atomic<i32>
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let id = global_id.x;

            if(id>=numParticles) {return;}

            var n = getNormal(id);        
            n.x = 0.0;
            n.y = 0.0;
            n.z = 0.0;

            let tempX = atomicLoad(&tempNormal[id * 3 + 0].value);
            let tempY = atomicLoad(&tempNormal[id * 3 + 1].value);
            let tempZ = atomicLoad(&tempNormal[id * 3 + 2].value);

            atomicStore(&tempNormal[id * 3 + 0].value, i32(0));
            atomicStore(&tempNormal[id * 3 + 1].value, i32(0));
            atomicStore(&tempNormal[id * 3 + 2].value, i32(0));

            n.x = f32(tempX) / 100.0;
            n.y = f32(tempY) / 100.0;
            n.z = f32(tempZ) / 100.0;

            n = normalize(n);

            normal[id*3 + 0] = f32(n.x);
            normal[id*3 + 1] = f32(n.y);
            normal[id*3 + 2] = f32(n.z);
        }
    `;

    getNormalSummationComputeShader(){
        return this.normalSummationComputeShader;
    }
}