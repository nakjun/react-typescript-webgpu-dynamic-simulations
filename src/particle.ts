export class Particle {
    // position: [number, number, number];
    // color: [number, number, number];

    // constructor(position: [number, number, number], color: [number, number, number]) {
    //     this.position = position;
    //     this.color = color;
    // }

    position: [number, number, number];
    color: [number, number, number];
    velocity: [number, number, number];

    constructor(position: [number, number, number], color: [number, number, number], velocity: [number, number, number] = [0, 0, 0]) {
        this.position = position;
        this.color = color;
        this.velocity = velocity;
    }

    // Add methods for updating particle logic if needed
}