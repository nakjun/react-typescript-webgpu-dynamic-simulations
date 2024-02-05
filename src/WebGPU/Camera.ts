import { mat4, vec3 } from 'gl-matrix';

export class Camera {
    position: vec3;
    target: vec3;
    up: vec3;
    fov: number; // Field of view, in radians
    aspectRatio: number;
    near: number;
    far: number;

    constructor(position: vec3, target: vec3, up: vec3, fov: number, aspectRatio: number, near: number, far: number) {
        this.position = position;
        this.target = target;
        this.up = up;
        this.fov = fov;
        this.aspectRatio = aspectRatio;
        this.near = near;
        this.far = far;
    }

    getViewMatrix(): mat4 {
        const view = mat4.create();
        mat4.lookAt(view, this.position, this.target, this.up);
        return view;
    }

    getProjectionMatrix(): mat4 {
        const projection = mat4.create();
        mat4.perspective(projection, this.fov, this.aspectRatio, this.near, this.far);
        return projection;
    }
}
