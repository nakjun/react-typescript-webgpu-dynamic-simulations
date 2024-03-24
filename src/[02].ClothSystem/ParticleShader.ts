export class ParticleShader {
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
            output.Color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Pass vertex color to fragment shader
            return output;
        }

        @fragment
        fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            let lightDir = normalize(vec3<f32>(0.0, 20.0, -1.0));
            let ambient = 1.0;
            let lighting = ambient + (1.0 - ambient); 
            let color = vec3<f32>(in.Color.xyz);
            return vec4<f32>(color * lighting, 1.0);
        }
    `;

    springShader = `
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
            output.Color = vec4<f32>(0.0, 1.0, 0.0, 1.0); // Pass vertex color to fragment shader
            return output;
        }

        @fragment
        fn fs_main(in: FragmentOutput) -> @location(0) vec4<f32> {
            let lightDir = normalize(vec3<f32>(0.0, 20.0, 5.0));
            let ambient = 0.1;
            let lighting = ambient + (1.0 - ambient); 
            let color = vec3<f32>(in.Color.xyz);
            return vec4<f32>(color * lighting, 1.0);        
        }
    `;

    textureShader = `
    struct TransformData {
        model: mat4x4<f32>,
        view: mat4x4<f32>,
        projection: mat4x4<f32>,
    };
    @binding(0) @group(0) var<uniform> transformUBO: TransformData;
    @binding(1) @group(0) var myTexture: texture_2d<f32>;
    @binding(2) @group(0) var mySampler: sampler;
    @binding(3) @group(0) var<uniform> cameraPos: vec3<f32>;    
    
    struct LightData {
        position: vec3<f32>,
        color: vec4<f32>,
        intensity: f32,
        specularStrength: f32,
        shininess: f32,
    };
    @binding(4) @group(0) var<uniform> lightUBO: LightData; // Light 정보 바인딩
    @binding(5) @group(0) var<uniform> alpha: f32;
    
    struct VertexOutput {
        @builtin(position) Position : vec4<f32>,
        @location(0) TexCoord : vec2<f32>,
        @location(1) Normal : vec3<f32>,
        @location(2) FragPos : vec3<f32>, // 프래그먼트 위치 추가
    };
    
    @vertex
    fn vs_main(@location(0) vertexPosition: vec3<f32>, @location(1) vertexTexCoord: vec2<f32>, @location(2) vertexNormal: vec3<f32>) -> VertexOutput {
        var output : VertexOutput;
        output.Position = transformUBO.projection * transformUBO.view * transformUBO.model * vec4<f32>(vertexPosition, 1.0);
        output.TexCoord = vertexTexCoord;
        output.Normal = (transformUBO.model * vec4<f32>(vertexNormal, 0.0)).xyz;
        output.FragPos = (transformUBO.model * vec4<f32>(vertexPosition, 1.0)).xyz; // 월드 공간 위치 계산
        return output;
    }
    
    @fragment
    fn fs_main(@location(0) TexCoord : vec2<f32>, @location(1) Normal : vec3<f32>, @location(2) FragPos: vec3<f32>) -> @location(0) vec4<f32> {            
        // let ambientStrength: f32 = 0.001; // 환경광 강도를 적절히 조절
        // let ambientColor: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0) * ambientStrength; // 환경광 색상을 자연스러운 톤으로 조정
        // var lightPos: vec3<f32> = lightUBO.position;
        // let lightColor: vec4<f32> = lightUBO.color;
        // let lightIntensity: f32 = lightUBO.intensity;

        // let texColor: vec4<f32> = textureSample(myTexture, mySampler, TexCoord);            
        // let norm: vec3<f32> = normalize(Normal);
        // let viewDir: vec3<f32> = normalize(cameraPos - FragPos);
        // var finalColor:vec4<f32> = ambientColor;

        // for(var i=0;i<2;i=i+1){
        //     if(i==1){
        //         lightPos = vec3<f32>(0.0, 1.0, 0.0);
        //     }
            
        //     let lightDir: vec3<f32> = normalize(lightPos - FragPos);
        //     let diff: f32 = max(dot(norm, lightDir), 0.0);
        //     let diffuse: vec4<f32> = lightColor * texColor * diff * lightIntensity; // 난반사 강도를 조정하여 디테일 강화
            
            
        //     let reflectDir: vec3<f32> = reflect(-lightDir, norm);
        //     let spec: f32 = pow(max(dot(viewDir, reflectDir), 0.0), lightUBO.shininess);
        //     let specular: vec4<f32> = lightColor * spec * lightUBO.specularStrength;
            
        //     finalColor = finalColor + diffuse + specular;
        // }
        
        // finalColor.a = 0.8; // 텍스처의 알파 값을 최종 색상의 알파 값으로 설정
        // return finalColor;

        let lightPos: vec3<f32> = lightUBO.position;
        let lightColor: vec4<f32> = lightUBO.color;
        let lightIntensity: f32 = lightUBO.intensity;
        
        // 주변광 계산
        //let ambientColor: vec4<f32> = vec4<f32>(0.513725, 0.435294, 1.0, 1.0) * 0.001;
        let ambientColor: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0) * 0.001;
    
        // // diffuse 계산
        let norm: vec3<f32> = normalize(Normal);
        let lightDir: vec3<f32> = normalize(lightPos - FragPos);
        let diff: f32 = max(dot(norm, lightDir), 0.0);
        let diffuse: vec4<f32> = lightColor * diff * lightIntensity * vec4<f32>(0.88, 0.88, 0.88, 1.0);
    
        // // specular 계산
        let viewDir: vec3<f32> = normalize(cameraPos - FragPos);
        let reflectDir: vec3<f32> = reflect(-lightDir, norm);
        let spec: f32 = pow(max(dot(viewDir, reflectDir), 0.0), lightUBO.shininess);
        let specular: vec4<f32> = lightColor * spec * vec4<f32>(0.729134, 0.729134, 0.729134, 1.0);

        // // 최종 색상 계산
        var finalColor: vec4<f32> = ambientColor + diffuse + specular;

        return vec4<f32>(finalColor.x,finalColor.y, finalColor.z, 1.0);
        
        //return vec4<f32>(diffuse.x, diffuse.y, diffuse.z, 1.0);
    }

    `;

    freefallComputeShader = `
    
    @group(0) @binding(0) var<storage, read_write> positions: array<f32>;
    @group(0) @binding(1) var<storage, read_write> velocities: array<f32>;    
    @group(0) @binding(2) var<storage, read_write> fixed: array<u32>;
    @group(0) @binding(3) var<storage, read_write> force: array<f32>;   
    @group(0) @binding(4) var<storage, read_write> prevPosition: array<f32>;
    @group(0) @binding(5) var<uniform> externalForce : vec3<f32>;    

    fn getPosition(index:u32) -> vec3<f32>{
        return vec3<f32>(positions[index*3],positions[index*3+1],positions[index*3+2]);
    }
    
    fn getVelocity(index:u32) -> vec3<f32>{
        return vec3<f32>(velocities[index*3],velocities[index*3+1],velocities[index*3+2]);
    }

    fn getForce(index:u32) -> vec3<f32>{
        return vec3<f32>(force[index*3] / 2.0,force[index*3+1] / 2.0,force[index*3+2] / 2.0);
    }

    @compute 
    @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        var fix = fixed[index];
        
        if(fix==1){
            return;
        }
        
        var pos = getPosition(index);
        var vel = getVelocity(index);
        var f = getForce(index);     
        
        prevPosition[index*3 + 0] = pos.x;
        prevPosition[index*3 + 1] = pos.y;
        prevPosition[index*3 + 2] = pos.z;
        var flag:bool = false;
        if(externalForce.z!=0.0){                                    
            var origin_pos = getPosition(0);
            var dist = distance(origin_pos, pos);
            if(dist < 20.0)
            {
                vel *= 0.3;
                f *= 0.0;
                if(pos.y >= -20.0){
                    pos.y -= 0.01;
                }                
                pos.z += 0.05;
                flag = true;
            }

            if(pos.z >= 150.0){
                fixed[index] = u32(1);
            }
        }

        if(flag==false){
            
            // var origin_location:vec3<f32> = vec3<f32>(0.0,0.0,0.0);
    
            // if(distance(pos, origin_location) < 20.0){
            //     var dir = normalize(origin_location-pos);
            //     pos += (-dir*0.25);
            //     vel *= 0.3;
            // }
            
            //floor collisions
            if(pos.y < -10.0){
                pos.y += 0.01;  
                vel *= -0.3;      
            }
        }
        
        var gravity: vec3<f32> = vec3<f32>(0.0, -9.8, 0.0);        
        var deltaTime: f32 = 0.001; // Assuming 60 FPS for simplicity
        vel += ((f + gravity) * deltaTime);
        pos += (vel * deltaTime);
        
        velocities[index*3 + 0] = vel.x;
        velocities[index*3 + 1] = vel.y;
        velocities[index*3 + 2] = vel.z;

        positions[index*3 + 0] = pos.x;
        positions[index*3 + 1] = pos.y;
        positions[index*3 + 2] = pos.z;
    }    
    `;

    getParticleShader() {
        return this.particle_shader;
    }

    getComputeShader() {
        return this.freefallComputeShader;
    }

    getSpringShader() {
        return this.springShader;
    }

    getTextureShader() {
        return this.textureShader;
    }
}
