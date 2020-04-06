from util import show_image
from flowermodel import FlowerModel

a = FlowerModel()
a.pre_processing()
a.train(5)
a.predict("./TestImages/rainbowRose.jpg")
show_image('./TestImages/rainbowRose.jpg')
a.saveModel("flowerData.pth")
