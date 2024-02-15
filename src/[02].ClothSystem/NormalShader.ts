export class NormalShader{
    normalUpdateComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> triangleCalculationBuffer: array<triangles>;
    @group(0) @binding(2) var<storage, read_write> tempNormal: array<f32>;
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
        t1: f32,
        t2: f32,
        t3: f32,
    }

    @compute @workgroup_size(64)
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

        var t1: u32 = u32(triangle.t1);
        var t2: u32 = u32(triangle.t2);
        var t3: u32 = u32(triangle.t3);

        // Fetching positions from the positions buffer
        var p0: vec3<f32> = getPosition(v1);
        var p1: vec3<f32> = getPosition(v2);
        var p2: vec3<f32> = getPosition(v3);

        // Calculate the normal for this triangle
        var normal: vec3<f32> = calculateNormal(p0, p1, p2);

        // The accumulation of normals should ideally be atomic or handled in a way that
        // ensures correct concurrent access; here, we simply assign it for demonstration.
        // In practice, you'll need to accumulate these in a way that prevents race conditions.
        tempNormal[t1 * 3 + 0] = normal.x;
        tempNormal[t1 * 3 + 1] = normal.y;
        tempNormal[t1 * 3 + 2] = normal.z;

        tempNormal[t2 * 3 + 0] = normal.x;
        tempNormal[t2 * 3 + 1] = normal.y;
        tempNormal[t2 * 3 + 2] = normal.z;

        tempNormal[t3 * 3 + 0] = normal.x;
        tempNormal[t3 * 3 + 1] = normal.y;
        tempNormal[t3 * 3 + 2] = normal.z;
    }
    `;

    getNormalUpdateComputeShader(){
        return this.normalUpdateComputeShader;
    }

    normalSummationComputeShader = `
        @group(0) @binding(0) var<storage, read_write> tempNormal: array<f32>;
        @group(0) @binding(1) var<storage, read_write> normal: array<f32>;
        @group(0) @binding(2) var<uniform> maxConnectedTriangle: u32;
        @group(0) @binding(3) var<uniform> numParticles: u32;

        fn getNormal(index:u32) -> vec3<f32>{
            return vec3<f32>(normal[index*3],normal[index*3+1],normal[index*3+2]);
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let id = global_id.x;

            if(id>=numParticles) {return;}

            var n = getNormal(id);        
            n.x = 0.0;
            n.y = 0.0;
            n.z = 0.0;

            var start = (id * maxConnectedTriangle);
            var end = (id * maxConnectedTriangle) + maxConnectedTriangle;

            for (var i: u32 = start; i < end; i++) {            
                n.x += (tempNormal[i * 3 + 0]);
                n.y += (tempNormal[i * 3 + 1]);
                n.z += (tempNormal[i * 3 + 2]);
            }

            n = normalize(n);
            //var normalizeData = normalize(n);

            normal[id*3 + 0] = f32(start);
            normal[id*3 + 1] = f32(end);
            normal[id*3 + 2] = f32(maxConnectedTriangle);
        }
    `;

    getNormalSummationComputeShader(){
        return this.normalSummationComputeShader;
    }
}