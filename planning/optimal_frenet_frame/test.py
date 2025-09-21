import numpy as np
from vispy import scene, app

# 캔버스 생성
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()

# 데이터 준비
x = np.linspace(0, 10, 100)
y = np.sin(x)
points = np.column_stack([x, y])

# Line 시각화 추가
line = scene.Line(points, parent=view.scene, color='blue')

# 카메라 설정 (2D plot 모드)
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range(x=(0, 10), y=(-1.5, 1.5))

app.run()

