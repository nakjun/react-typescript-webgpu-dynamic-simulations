import { Renderer } from "./Renderer";


export const Initialize = async () => {    
  const sceneManager = new Renderer("gfx-main");
  sceneManager.init().then(() => {    
    sceneManager.createBuffers();
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
}