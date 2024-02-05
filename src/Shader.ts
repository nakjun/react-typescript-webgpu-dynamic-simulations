export class Shader {
    render_shader = `struct TransformData {
                    model: mat4x4<f32>,
                    view: mat4x4<f32>,
                    projection: mat4x4<f32>,
                };
                @group(0) @binding(0) var<uniform> transformUBO: TransformData;

            struct VertexInput {
            @location(0) position : vec3<f32>
            };

            struct FragmentOutput {
            @builtin(position) Position : vec4<f32>,
            @location(0) Color : vec4<f32>
            };

            @vertex
            fn vs_main(vertexInput: VertexInput) -> FragmentOutput {
            var output : FragmentOutput;
            let modelViewProj = transformUBO.projection * transformUBO.view * transformUBO.model;
            output.Position = modelViewProj * vec4<f32>(vertexInput.position, 1.0);
            output.Color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Fixed color for all particles

            return output;
            }

            @fragment
            fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            return in.Color;
            }

            `;
    


    getRenderShader(){
        return this.render_shader;
    }
}