export class Shader {    
    particle_shader = `
        struct TransformData {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> transformUBO: TransformData;

        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };

        struct FragmentOutput {
            @builtin(position) Position: vec4<f32>,
            @location(0) Color: vec4<f32>,
        };

        @vertex
        fn vs_main(vertexInput: VertexInput) -> FragmentOutput {
            var output: FragmentOutput;
            let modelViewProj = transformUBO.projection * transformUBO.view * transformUBO.model;
            output.Position = modelViewProj * vec4<f32>(vertexInput.position, 1.0);
            output.Color = vec4<f32>(vertexInput.color, 1.0); // Pass vertex color to fragment shader
            return output;
        }

        @fragment
        fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            let lightDir = normalize(vec3<f32>(0.0, 0.0, -1.0));
            let ambient = 0.1;
            let lighting = ambient + (1.0 - ambient); 
            let color = vec3<f32>(in.Color.xyz);
            return vec4<f32>(color * lighting, 1.0);        
        }
    `;

    freefallComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
    @group(0) @binding(2) var<uniform> numParticles: i32;
    
    // Simple collision response that inverts velocity upon collision
    fn respondToCollision(velocity: vec3<f32>) -> vec3<f32> {
        // Invert velocity for simplicity; replace with your collision response
        return -velocity;
    }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        var pos = positions[index];
        var vel = velocities[index];

        let gravity: vec3<f32> = vec3<f32>(0.0, -9.8, 0.0);
        var deltaTime: f32 = 0.002; // Assuming 60 FPS for simplicity
        
        if(pos.y < 0.0) {
            pos.y = 0.01;
            vel.y *= -0.95;
        }

        vel += gravity * deltaTime;
        pos += velocities[index] * deltaTime;

        velocities[index] = vel;
        positions[index] = pos;
    }
    
    `;

    getParticleShader(){
        return this.particle_shader;
    }

    getComputeShader(){
        return this.freefallComputeShader;
    }
}
