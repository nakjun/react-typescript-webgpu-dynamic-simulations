import { Particle } from "./particle";
import { WebGPURenderer } from "./WebGPURenderer";
import { Renderer } from "./Render";


export const Initialize = async () => {  
  const sceneManager = new WebGPURenderer("gfx-main");
  sceneManager.init().then(() => {
    sceneManager.createParticles(5); // Create 100 particles
    sceneManager.createBuffers();
    sceneManager.createComputePipeline();
    sceneManager.createPipeline();   
    animate();
  });

  // Create an animation loop function
  function animate() {
    // Update the particle simulation (e.g., call your compute shader)
    //sceneManager.dispatchComputeShader();

    // Render the scene (e.g., call your render function)
    sceneManager.render();

    // Request the next frame
    requestAnimationFrame(animate);
  }

  // const canvas : HTMLCanvasElement = <HTMLCanvasElement> document.getElementById("gfx-main");

  // const renderer = new Renderer(canvas);

  // renderer.Initialize();

}