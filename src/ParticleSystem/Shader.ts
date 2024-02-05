export class Shader {
    render_shader = `
        struct TransformData {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> transformUBO: TransformData;

        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>, // Add this line to accept color per vertex
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
            return in.Color; // Use the color passed from the vertex shader
        }
    `;

    particle_shader = `
        struct TransformData {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> transformUBO: TransformData;

        struct VertexInput {
            @location(0) position: vec3<f32>
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
            output.Color = vec4<f32>(1.0, 1.0, 1.0, 1.0); // Pass vertex color to fragment shader
            return output;
        }

        @fragment
        fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            return in.Color; // Use the color passed from the vertex shader
        }
    `;

    freefallComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
    
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        var pos = positions[index];
        var vel = velocities[index];

        let gravity: vec3<f32> = vec3<f32>(0.0, -9.8, 0.0);
        let deltaTime: f32 = 0.005; // Assuming 60 FPS for simplicity
        
        if(pos.y < 0.0) {
            pos.y = 0.0001;
            vel *= -0.95;
        }

        vel += gravity * deltaTime;
        pos += velocities[index] * deltaTime;

        velocities[index] = vel;
        positions[index] = pos;
    }
    
    `;

    getRenderShader(){
        return this.render_shader;
    }

    getParticleShader(){
        return this.particle_shader;
    }

    getComputeShader(){
        return this.freefallComputeShader;
    }
}
