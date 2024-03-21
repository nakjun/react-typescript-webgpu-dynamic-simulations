export class SpringShader {

    springUpdateShader = `
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
    @group(0) @binding(2) var<storage, read> springs: array<f32>;
    @group(0) @binding(3) var<uniform> numSprings: u32;
    @group(0) @binding(4) var<storage, read_write> nodeForce: array<atomicI32>;
    @group(0) @binding(5) var<uniform> numParticles: u32;

    struct atomicI32{
        value: atomic<i32>
    }

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }
    
    fn getVelocity(index:u32) -> vec3<f32>{
        return vec3<f32>(velocities[index*3],velocities[index*3+1],velocities[index*3+2]);
    }


    @compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let id = global_id.x;

    if(id >= numSprings){return;}

    var i1 = u32(springs[id * 5 + 0]);
    var i2 = u32(springs[id * 5 + 1]);

    var pos1 = getPosition(i1);
    var pos2 = getPosition(i2);

    var vel1 = getVelocity(i1);
    var vel2 = getVelocity(i2);

    if(length(pos2-pos1) < 0.00001){
        return;
    }

    var dir = normalize(pos2-pos1);
    var springForce = springs[id * 5 + 2] * (length(pos2-pos1) - springs[id * 5 + 4]);
    var dampingForce = springs[id * 5 + 3] * (dot(vel2, dir) - dot(vel1, dir));

    var estimatedForce = (springForce + dampingForce)*dir;

    atomicAdd(&nodeForce[i1 * 3 + 0].value, i32(estimatedForce.x*100.0));
    atomicAdd(&nodeForce[i1 * 3 + 1].value, i32(estimatedForce.y*100.0));
    atomicAdd(&nodeForce[i1 * 3 + 2].value, i32(estimatedForce.z*100.0));

    atomicAdd(&nodeForce[i2 * 3 + 0].value, i32(-estimatedForce.x*100.0));
    atomicAdd(&nodeForce[i2 * 3 + 1].value, i32(-estimatedForce.y*100.0));
    atomicAdd(&nodeForce[i2 * 3 + 2].value, i32(-estimatedForce.z*100.0));
}
    `;

    getSpringUpdateShader() {
        return this.springUpdateShader;
    }

    nodeForceSummation = `
    @group(0) @binding(0) var<storage, read_write> nodeForce: array<atomicI32>;
    @group(0) @binding(1) var<storage, read_write> force: array<f32>;
    @group(0) @binding(2) var<uniform> numParticles: u32;

    fn getForce(index:u32) -> vec3<f32>{
        return vec3<f32>(force[index*3],force[index*3+1],force[index*3+2]);
    }

    struct atomicI32{
        value: atomic<i32>
    }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let id = global_id.x;

        if(id>=numParticles) {return;}

        var f = getForce(id);

        let tempX = atomicLoad(&nodeForce[id * 3 + 0].value);
        let tempY = atomicLoad(&nodeForce[id * 3 + 1].value);
        let tempZ = atomicLoad(&nodeForce[id * 3 + 2].value);

        f.x = f32(tempX) / 100.0;
        f.y = f32(tempY) / 100.0;
        f.z = f32(tempZ) / 100.0;

        force[id*3 + 0] = f32(f.x);
        force[id*3 + 1] = f32(f.y);
        force[id*3 + 2] = f32(f.z);
    }

    @compute @workgroup_size(256)
    fn initialize(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let id = global_id.x;

        if(id>=numParticles) {return;}
        
        atomicStore(&nodeForce[id * 3 + 0].value, i32(0));
        atomicStore(&nodeForce[id * 3 + 1].value, i32(0));
        atomicStore(&nodeForce[id * 3 + 2].value, i32(0));
        
        force[id*3 + 0] = f32(0.0);
        force[id*3 + 1] = f32(0.0);
        force[id*3 + 2] = f32(0.0);
    }
    
    `;

    getNodeForceShader(){
        return this.nodeForceSummation;
    }
}