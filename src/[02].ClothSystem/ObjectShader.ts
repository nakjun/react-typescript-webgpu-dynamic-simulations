export class ObjectShader{
    moveComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }

    @compute 
    @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;        
        var pos = getPosition(index);        
        
        pos.x -= 0.005;

        positions[index*3 + 0] = pos.x;
        positions[index*3 + 1] = pos.y;
        positions[index*3 + 2] = pos.z;
    }    
    `;

    getMoveShader(){
        return this.moveComputeShader;
    }
}