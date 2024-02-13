import GUI from 'lil-gui';

export class SystemGUI{
    performanceGui!: GUI;

    constructor(){
        this.performanceGui = new GUI({
            container: document.body,
            autoPlace: false // 기본 위치 배치 사용 안함
        });
        this.performanceGui.title("Performance");
        this.performanceGui.domElement.style.position = 'absolute';
        this.performanceGui.domElement.style.top = '0px';
        this.performanceGui.domElement.style.left = '0px';        
    }
}