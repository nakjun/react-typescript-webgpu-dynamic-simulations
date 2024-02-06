export class SpringShader{
    
    springUpdateShader = `
    @group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
    @group(0) @binding(2) var<storage, read> springs: array<Spring>; // Define SpringData appropriately
    @group(0) @binding(3) var<uniform> numSprings: u32;

    struct Spring {
        index1: u32,
        index2: u32,
        ks: f32,
        kd: f32,
        mRestLen: f32
    };

    @compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let id = global_id.x;
    if (id < numSprings) {
        let spring = springs[id];
        let pos1 = positions[spring.index1];
        let pos2 = positions[spring.index2];
        let vel1 = velocities[spring.index1];
        let vel2 = velocities[spring.index2];

        let posDirection = pos2 - pos1;
        let velDirection = vel2 - vel1;

        let len = length(posDirection);
        let forceDirection = normalize(posDirection);
        let spforce = (len - spring.mRestLen) * spring.ks;    
        let damp = dot(velDirection, forceDirection) * spring.kd / len;    
    }
}
    `;

    getSpringUpdateShader(){
        return this.springUpdateShader;
    }
}