export class ObjModel {
    vertices: number[] = [];
    indices: number[] = [];
    normals: number[] = [];
}

export class ObjLoader {
    parse(objData: string): ObjModel {
        const model = new ObjModel();
        const vertexPositions: number[][] = [];
        const vertexNormals: number[][] = [];
        const tempNormals: number[][] = [];
        const faces: string[][] = [];

        objData.split('\n').forEach(line => {
            const parts = line.trim().split(/\s+/);
            switch (parts[0]) {
                case 'v':
                    vertexPositions.push(parts.slice(1).map(parseFloat));
                    tempNormals.push([0, 0, 0]); // 임시 법선 배열 초기화
                    break;
                case 'vn':
                    vertexNormals.push(parts.slice(1).map(parseFloat));
                    break;
                case 'f':
                    faces.push(parts.slice(1));
                    break;
            }
        });

        const indexMap = new Map<string, number>();
        let currentIndex = 0;

        faces.forEach(faceParts => {
            const faceIndices = faceParts.map(part => {
                const [pos, tex, norm] = part.split('/').map(e => parseInt(e) - 1);
                const key = `${pos}|${norm}`;

                if (indexMap.has(key)) {
                    return indexMap.get(key)!;
                } else {
                    const position = vertexPositions[pos];
                    model.vertices.push(...position);

                    if (vertexNormals.length > 0 && norm !== undefined) {
                        const normal = vertexNormals[norm];
                        model.normals.push(...normal);
                    }

                    indexMap.set(key, currentIndex);
                    return currentIndex++;
                }
            });

            for (let i = 1; i < faceIndices.length - 1; i++) {
                model.indices.push(faceIndices[0], faceIndices[i], faceIndices[i + 1]);
            }
        });

        if (vertexNormals.length === 0) {
            this.calculateNormals(model.vertices, model.indices).forEach(n => model.normals.push(n));
        }

        return model;
    }

    calculateNormals(vertices: number[], indices: number[]): number[] {
        const normals = new Array(vertices.length).fill(0);

        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i] * 3;
            const i1 = indices[i + 1] * 3;
            const i2 = indices[i + 2] * 3;

            const v1 = vertices.slice(i0, i0 + 3);
            const v2 = vertices.slice(i1, i1 + 3);
            const v3 = vertices.slice(i2, i2 + 3);

            const u = v2.map((val, idx) => val - v1[idx]);
            const v = v3.map((val, idx) => val - v1[idx]);

            const norm = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]
            ];

            norm.forEach((value, idx) => {
                normals[i0 + idx] += value;
                normals[i1 + idx] += value;
                normals[i2 + idx] += value;
            });
        }

        for (let i = 0; i < normals.length; i += 3) {
            const len = Math.sqrt(normals[i] ** 2 + normals[i + 1] ** 2 + normals[i + 2] ** 2);
            normals[i] /= len;
            normals[i + 1] /= len;
            normals[i + 2] /= len;
        }

        return normals;
    }

    async load(url: string): Promise<ObjModel> {
        const response = await fetch(url);
        const objData = await response.text();
        return this.parse(objData);
    }
}
