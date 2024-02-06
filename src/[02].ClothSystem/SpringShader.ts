export class SpringShader{
    
    springUpdateShader = `
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
    @group(0) @binding(2) var<storage, read> springs: array<Spring>; // Define SpringData appropriately
    @group(0) @binding(3) var<uniform> numSprings: u32;

    struct Spring {
        index1: u32,
        index2: u32,
        ks: f32,
        kd: f32,
        mRestLen: f32
    };

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }
    
    fn getVelocity(index:u32) -> vec3<f32>{
        return vec3<f32>(velocities[index*3],velocities[index*3+1],velocities[index*3+2]);
    }


    @compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let id = global_id.x;
    if (id < numSprings) {
        let spring = springs[id];
        let pos1 = getPosition(spring.index1);
        let pos2 = getPosition(spring.index2);
        let vel1 = getVelocity(spring.index1);
        let vel2 = getVelocity(spring.index2);

        let posDirection = pos2 - pos1;
        let velDirection = vel2 - vel1;

        let len = length(posDirection);
        let forceDirection = normalize(posDirection);
        let spforce = (len - spring.mRestLen) * spring.ks;    
        let damp = dot(velDirection, forceDirection) * spring.kd / len;    

        let estimatedForce = (spforce+damp) / len;
    }
}
    `;

    getSpringUpdateShader(){
        return this.springUpdateShader;
    }
}