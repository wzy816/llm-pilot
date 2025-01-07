import hanlp

print(hanlp.pretrained.mtl)

en_text = """
The image depicts a city street at night, with a wide, two-lane road that is empty except for a few cars in the distance. The street is lined with trees and buildings on either side, and there are streetlights illuminating the road.

* A city street at night with a wide, two-lane road:
        + The road is empty except for a few cars in the distance.
        + The street is lined with trees and buildings on either side.
        + There are streetlights illuminating the road.
* The street is lined with trees and buildings on either side:
        + The trees are tall and have green leaves.
        + The buildings are tall and have windows that reflect the streetlights.
        + There are also some smaller buildings and signs along the street.
* There are streetlights illuminating the road:
        + The streetlights are tall and have a warm glow.
        + They are spaced evenly apart along the length of the street.
        + They provide enough light to illuminate the road and the surrounding area.

Overall, the image suggests that the city is well-lit and safe at night, with a wide and empty road that is easy to navigate. The presence of trees and buildings along the street adds to the urban feel of the image.
"""

zh_text = """
这张图片展示了一条夜晚的城市街道。以下是一些识别的物体：

1. 道路：一条宽阔的马路，中间有分隔带。
2. 车辆：几辆汽车在道路上行驶，尾灯亮着，显示出它们正在移动。
3. 路灯：路边的路灯亮着，照亮了道路。
4. 建筑物：背景中有几栋高楼大厦，灯光亮着，显示出城市的繁华。
5. 树木：路边有一些树木，树叶在灯光的照射下显得清晰。
6. 广告牌：远处的广告牌亮着，显示着一些信息。

这些物体共同构成了一个繁忙的城市夜晚景象。

"""
tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)

ans = tok_fine(en_text) + tok_fine(zh_text)
print(ans)
