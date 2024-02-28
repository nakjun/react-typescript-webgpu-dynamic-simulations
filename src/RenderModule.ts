import { Renderer } from "./[01].ParticleSystem/Renderer";
import { ClothRenderer } from "./[02].ClothSystem/Renderer";
import { ObjModel, ObjLoader } from "./Common/ObjLoader";

const startParticleSimulation = async () => {
  const sceneManager = new Renderer("gfx-main");
  sceneManager.init().then(() => {
    sceneManager.createBuffers();
    sceneManager.createPipeline();
    sceneManager.createParticles(100000);
    sceneManager.createParticleBuffers();
    sceneManager.createParticlePipeline();
    sceneManager.createComputePipeline();
    sceneManager.createComputeBindGroup();
    animate();
  });

  function animate() {
    sceneManager.render();
    requestAnimationFrame(animate);
  }
}

const startClothSimluation = async () => {

  const canvas = document.querySelector("canvas#gfx-main") as HTMLCanvasElement; // `as HTMLCanvasElement`로 타입 단언 사용
  if (!canvas) {
    console.error("Canvas element not found");
    return;
  }

  let isLeftMouseDown = false;
  let isRightMouseDown = false;
  let lastMouseX: number, lastMouseY: number;

  canvas.addEventListener('mousedown', (event: MouseEvent) => { // `MouseEvent` 타입 명시
    if (event.button === 0) { // 좌클릭
      isLeftMouseDown = true;
      console.log("좌클릭");
    } else if (event.button === 2) { // 우클릭
      isRightMouseDown = true;
      console.log("우클릭");
    }
    lastMouseX = event.clientX;
    lastMouseY = event.clientY;

  });

  document.addEventListener('mouseup', (event) => {
    isLeftMouseDown = false;
    isRightMouseDown = false;
  });



  const sceneManager = new ClothRenderer("gfx-main");
  sceneManager.init().then(() => {
    canvas.addEventListener('mousemove', (event) => {
      if (isLeftMouseDown) {
        // 카메라 회전 로직 구현
        const dx = event.clientX - lastMouseX;
        const dy = event.clientY - lastMouseY;
        //console.log("rotate");
        sceneManager.rotateCamera(dx, dy);

      } else if (isRightMouseDown) {
        // 카메라 패닝 로직 구현
        const dx = event.clientX - lastMouseX;
        const dy = event.clientY - lastMouseY;

        // 패닝 로직 실행
        sceneManager.panCamera(dx, dy);
      }
      lastMouseX = event.clientX;
      lastMouseY = event.clientY;
    });

    canvas.addEventListener('wheel', (event) => {
      // 카메라 줌 인/아웃 로직 구현
      sceneManager.zoomCamera(event.deltaY / 100);
    });
    //sceneManager.createClothModel(2, 2, 200.0, 150.0, 1000.0, 0.5);  
    //sceneManager.createClothModel(3, 3, 500.0, 250.0, 1500.0, 0.3);
    //sceneManager.createClothModel(80, 80, 5000.0, 1550.0, 100000.0, 0.1);
    //sceneManager.createClothModel(200, 200, 10000.0, 3500.0, 65000.0, 0.1);    
    sceneManager.createClothModel(400, 400, 15000.0, 5000.0, 95000.0, 0.001);
    //sceneManager.createClothModel(750, 750, 7000.0, 5000.0, 20000.0, 0.03);
    //sceneManager.createClothModel(850, 850, 15000.0, 9500.0, 70000.0, 0.001);
    sceneManager.createClothBuffers();
    sceneManager.createRenderPipeline();
    sceneManager.createSpringPipeline();
    sceneManager.createTrianglePipeline();
    sceneManager.createParticlePipeline();
    sceneManager.createUpdateNormalPipeline();
    sceneManager.createSpringForceComputePipeline();
    sceneManager.createNodeForceSummationPipeline();
    sceneManager.createIntersectionPipeline();
    sceneManager.createTriTriIntersectionPipeline();

    animate();
  });

  // Create an animation loop function
  function animate() {
    sceneManager.render();
    requestAnimationFrame(animate);
  }
}

export const Initialize = async () => {

  var isCloth: boolean = true;

  if (!isCloth) {
    await startParticleSimulation();
  } else {



    await startClothSimluation();
  }



}