export class SpringShader {

    springUpdateShader = `
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
    @group(0) @binding(2) var<storage, read> springs: array<Spring>; // Define SpringData appropriately    
    @group(0) @binding(3) var<uniform> numSprings: u32;
    @group(0) @binding(4) var<storage, read_write> nodeForce: array<f32>; // Define SpringData appropriately
    @group(0) @binding(5) var<uniform> numParticles: u32;

    struct Spring {
        index1: u32,
        index2: u32,
        ks: f32,
        kd: f32,
        mRestLen: f32,
        target1: u32,
        target2: u32,
    };

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }
    
    fn getVelocity(index:u32) -> vec3<f32>{
        return vec3<f32>(velocities[index*3],velocities[index*3+1],velocities[index*3+2]);
    }


    @compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let id = global_id.x;
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

        let estimatedForce = forceDirection * (spforce+damp) / len;

        let target1 = (numParticles-1) * spring.index1 + spring.target1;
        nodeForce[target1*3 + 0] = estimatedForce.x;
        nodeForce[target1*3 + 1] = estimatedForce.y;
        nodeForce[target1*3 + 2] = estimatedForce.z;
        
        let target2 = (numParticles-1) * spring.index2 + spring.target2;
        nodeForce[target2*3 + 0] = -estimatedForce.x;
        nodeForce[target2*3 + 1] = -estimatedForce.y;
        nodeForce[target2*3 + 2] = -estimatedForce.z;
}
    `;

    getSpringUpdateShader() {
        return this.springUpdateShader;
    }

    nodeForceSummation = `
        @group(0) @binding(0) var<storage, read_write> nodeForce: array<f32>; // Define SpringData appropriately
        @group(0) @binding(1) var<storage, read_write> Force: array<f32>; // Define SpringData appropriately
        @group(0) @binding(2) var<uniform> maxConnectedSpring: u32;
        @group(0) @binding(3) var<uniform> numParticles: u32;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let id = global_id.x;

            //initialize
            Force[id * 3 + 0] = 0;
            Force[id * 3 + 1] = 0;
            Force[id * 3 + 2] = 0;

            for (var i: u32 = 0; i < maxConnectedSpring; i++) {
                let newIndex = id * (numParticles-1) + i; 
                //summation
                Force[id * 3 + 0] += nodeForce[newIndex*3 + 0];
                Force[id * 3 + 1] += nodeForce[newIndex*3 + 1];
                Force[id * 3 + 2] += nodeForce[newIndex*3 + 2];
                
                //reset;
                nodeForce[newIndex*3 + 0] = 0.0;
                nodeForce[newIndex*3 + 1] = 0.0;
                nodeForce[newIndex*3 + 2] = 0.0;
            }
        }
    `;

    getNodeForceShader(){
        return this.nodeForceSummation;
    }
}