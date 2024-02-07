export class SpringShader {

    springUpdateShader = `
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
    @group(0) @binding(2) var<storage, read> springs: array<Spring>; // Define SpringData appropriately    
    @group(0) @binding(3) var<uniform> numSprings: u32;
    @group(0) @binding(4) var<storage, read_write> nodeForce: array<f32>;
    @group(0) @binding(5) var<uniform> numParticles: u32;

    struct Spring {
        index1: f32,
        index2: f32,
        ks: f32,
        kd: f32,
        mRestLen: f32,
        target1: f32,
        target2: f32,
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

    if(id >= numSprings){return;}

    var spring = springs[id];

    var i1 = u32(spring.index1);
    var i2 = u32(spring.index2);

    var t1 = u32(spring.target1);
    var t2 = u32(spring.target2);
    var pos1 = getPosition(i1);
    var pos2 = getPosition(i2);

    var vel1 = getVelocity(i1);
    var vel2 = getVelocity(i2);

    var posDirection = pos2 - pos1;
    var velDirection = vel2 - vel1;

    var len = length(posDirection);
    var forceDirection = normalize(posDirection);
    var spforce = (len - spring.mRestLen) * spring.ks;    
    var damp = dot(velDirection, forceDirection) / len * spring.kd;    

    var estimatedForce = forceDirection * (spforce+damp) / len;              

    nodeForce[t1 * 3 + 0] = estimatedForce.x;
    nodeForce[t1 * 3 + 1] = estimatedForce.y;
    nodeForce[t1 * 3 + 2] = estimatedForce.z;

    nodeForce[t2 * 3 + 0] = -estimatedForce.x;
    nodeForce[t2 * 3 + 1] = -estimatedForce.y;
    nodeForce[t2 * 3 + 2] = -estimatedForce.z;
}
    `;

    getSpringUpdateShader() {
        return this.springUpdateShader;
    }

    nodeForceSummation = `
    @group(0) @binding(0) var<storage, read_write> nodeForce: array<f32>;
    @group(0) @binding(1) var<storage, read_write> force: array<f32>;
    @group(0) @binding(2) var<uniform> maxConnectedSpring: u32;
    @group(0) @binding(3) var<uniform> numParticles: u32;

    fn getForce(index:u32) -> vec3<f32>{
        return vec3<f32>(force[index*3],force[index*3+1],force[index*3+2]);
    }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let id = global_id.x;

        if(id>=numParticles) {return;}

        var f = getForce(id);        
        f.x = 0.0;
        f.y = 0.0;
        f.z = 0.0;
        
        // 파티클별 힘 합산을 0으로 초기화
        // 각 파티클에 대해 연결된 모든 스프링의 힘을 합산

        var start = (id * maxConnectedSpring);
        var end = (id * maxConnectedSpring) + maxConnectedSpring;

        for (var i: u32 = start; i < end; i++) {            
            f.x += nodeForce[i * 3 + 0];
            f.y += nodeForce[i * 3 + 1];
            f.z += nodeForce[i * 3 + 2];
        }

        // force[id * 3 + 0] = f32(id * maxConnectedSpring);
        // force[id * 3 + 1] = f32(end);
        // force[id * 3 + 2] = f32(id);

        force[id*3 + 0] = f32(f.x);
        force[id*3 + 1] = f32(f.y);
        force[id*3 + 2] = f32(f.z);
    }

    @compute @workgroup_size(256)
    fn initialize(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let id = global_id.x;

        if(id>=numParticles) {return;}

        var f = getForce(id);     
           
        f.x = 0.0;
        f.y = 0.0;
        f.z = 0.0;
        
        force[id*3 + 0] = f32(f.x);
        force[id*3 + 1] = f32(f.y);
        force[id*3 + 2] = f32(f.z);
    }
    
    `;

    getNodeForceShader(){
        return this.nodeForceSummation;
    }
}