import { vec3 } from "gl-matrix";

export class Model {
    // Cube vertex positions (a simple cube centered at the origin)
    cubeVertices = new Float32Array([
        // Front face
        -1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        -1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        // Back face
        -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
        -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
        1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
        // Top face
        -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
        -1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
        // Bottom face
        -1.0, -1.0, -1.0, 0.0, 0.0, 1.0,
        1.0, -1.0, -1.0, 0.0, 0.0, 1.0,
        1.0, -1.0, 1.0, 0.0, 0.0, 1.0,
        -1.0, -1.0, 1.0, 0.0, 0.0, 1.0,
        // Right face
        1.0, -1.0, -1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, -1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, -1.0, 1.0, 0.0, 1.0, 1.0,
        // Left face
        -1.0, -1.0, -1.0, 0.25, 0.25, 0.25,
        -1.0, -1.0, 1.0, 0.25, 0.25, 0.25,
        -1.0, 1.0, 1.0, 0.25, 0.25, 0.25,
        -1.0, 1.0, -1.0, 0.25, 0.25, 0.25,
    ]);

    cubeColors = new Float32Array([
        
    ])

    cubeIndices = new Uint16Array([ 
        // Front
        0, 1, 2, 0, 2, 3,
        // Back
        4, 5, 6, 4, 6, 7,
        // Top
        8, 9, 10, 8, 10, 11,
        // Bottom
        12, 13, 14, 12, 14, 15,
        // Right
        16, 17, 18, 16, 18, 19,
        // Left
        20, 21, 22, 20, 22, 23
    ]);

    get_cubeVertices() {
        return this.cubeVertices;
    }

    get_cubeIndices() {
        return this.cubeIndices;
    }

    createSphere(radius:number, segments:number, position:vec3) {
        const vertices = [];
        const uvs = []; 
        const indices = [];        
        const normals = [];
    
        for (let lat = 0; lat <= segments; lat++) {
            const theta = lat * Math.PI / segments;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);
    
            for (let lon = 0; lon <= segments; lon++) {
                const phi = lon * 2 * Math.PI / segments;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);
    
                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;
                const u = 0.25 - (lon / segments);
                const v = 0.5 - (lat / segments);

                const nx = x;
                const ny = y;
                const nz = z;
                normals.push(nx, ny, nz); // 법선은 이미 정규화된 상태
    
                vertices.push(radius * x + position[0], radius * y+ position[1], radius * z+ position[2]);
                uvs.push(u, v);
            }
        }
    
        for (let lat = 0; lat < segments; lat++) {
            for (let lon = 0; lon < segments; lon++) {
                const first = (lat * (segments + 1)) + lon;
                const second = first + segments + 1;
    
                indices.push(first, second, first + 1);
                indices.push(second, second + 1, first + 1);
            }
        }
    
        return { vertices, uvs, indices, normals };
    }
}