from ImageData import ImageData
from ConvNet import ConvNet
import pandas as pd

<<<<<<< HEAD
def main() -> None:

    dat1 = ImageData(test_size=.75, gen_new_images=True)
    cn1 = ConvNet(cross_validating=True, model_num=16, conv_layers=(32, 64, 128, 256),
                  dense_layers=(256, 128), epochs=500, learning_rate=0.0001, patience=10)
    cn1.fit(dat1)
    sub1 = cn1.evaluate(dat1)

    dat1 = ImageData(test_size=.75, gen_new_images=True)
    cn2 = ConvNet(cross_validating=True, model_num=17, conv_layers=(32, 64, 128, 128),
                  dense_layers=(256, 128), epochs=500, learning_rate=0.0001, patience=10)
    cn2.fit(dat1)
    sub2 = cn2.evaluate(dat1)

    dat1 =ImageData(test_size=.75, gen_new_images=True)
    cn3 = ConvNet(cross_validating=True, model_num=18, conv_layers=(32, 64, 128, 256),
                  dense_layers=(512, 256), epochs=500, learning_rate=0.0001, patience=10)
    cn3.fit(dat1)
    sub3 = cn3.evaluate(dat1)

    dat1 = ImageData(test_size=.75, gen_new_images=True)
    cn4 = ConvNet(cross_validating=True, model_num=19, conv_layers=(16, 32, 64, 128),
                  dense_layers=(256, 128), epochs=500, learning_rate=0.0001, patience=10)
    cn4.fit(dat1)
    sub4 = cn4.evaluate(dat1)

    dat1 = ImageData(test_size=.75, gen_new_images=True)
    cn5 = ConvNet(cross_validating=True, model_num=20, conv_layers=(64, 128, 128, 64),
                  dense_layers=(256, 128), epochs=500, learning_rate=0.0001, patience=10)
    cn5.fit(dat1)
    sub5 = cn5.evaluate(dat1)

    

    ensemble = pd.DataFrame([sub1.id,
                             sub1.is_icerberg,
                             sub2.is_iceberg,
                             sub3.is_iceberg,
                             sub4.is_iceberg,
                             sub5.is_iceberg]).T
=======
def main():
    
    # base model with angle implemented
    dat1 = ImageData(train_size=.75, gen_new_images=False)
    cn1 = ConvNet(cross_validating=True, model_num=200, conv_layers=(64, 128, 128, 64),
                    dense_layers=(512, 256), epochs=500, learning_rate=0.001, dropout=0, patience=10)
    cn1.fit(dat1)
    sub1 = cn1.evaluate(dat1)

    # lower learning rate
    dat1 = ImageData(train_size=.75, gen_new_images=False)
    cn2 = ConvNet(cross_validating=True, model_num=201, conv_layers=(64, 128, 128, 64),
                    dense_layers=(512, 256), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn2.fit(dat1)
    sub2 = cn2.evaluate(dat1)

    # lower lr, gen new images
    dat1 =ImageData(train_size=.75, gen_new_images=True)
    cn3 = ConvNet(cross_validating=True, model_num=202, conv_layers=(64, 128, 128, 64),
                    dense_layers=(512, 256), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn3.fit(dat1)
    sub3 = cn3.evaluate(dat1)

    # lower lr, dropout 
    dat1 =ImageData(train_size=.75, gen_new_images=False)
    cn4 = ConvNet(cross_validating=True, model_num=203, conv_layers=(64, 128, 128, 64),
                    dense_layers=(512, 256), epochs=500, learning_rate=0.0001, dropout=.3, patience=10)
    cn4.fit(dat1)
    sub4 = cn4.evaluate(dat1)

    # set 1
    dat1 = ImageData(train_size=.75, gen_new_images=True)
    cn5 = ConvNet(cross_validating=True, model_num=204, conv_layers=(16, 32, 64, 128),
                  dense_layers=(256, 128), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn5.fit(dat1)
    sub5 = cn5.evaluate(dat1)

    dat1 = ImageData(train_size=.75, gen_new_images=True)
    cn6 = ConvNet(cross_validating=True, model_num=205, conv_layers=(16, 32, 64, 128),
                  dense_layers=(512, 256), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn6.fit(dat1)
    sub6 = cn6.evaluate(dat1)

>>>>>>> 980052548dd8be2efd48a7a7fd93b484b6a3b985

    dat1 = ImageData(train_size=.75, gen_new_images=True)
    cn8 = ConvNet(cross_validating=True, model_num=206, conv_layers=(32, 64, 128, 256),
                  dense_layers=(256, 128), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn8.fit(dat1)
    sub8 = cn8.evaluate(dat1)

    dat1 = ImageData(train_size=.75, gen_new_images=True)
    cn9 = ConvNet(cross_validating=True, model_num=207, conv_layers=(32, 64, 128, 256),
                  dense_layers=(512, 256), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn9.fit(dat1)
    sub9 = cn9.evaluate(dat1)

    
    dat1 = ImageData(train_size=.75, gen_new_images=True)
    cn10 = ConvNet(cross_validating=True, model_num=208, conv_layers=(32, 64, 128, 128),
                    dense_layers=(256, 128), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn10.fit(dat1)
    sub10 = cn10.evaluate(dat1)

    dat1 = ImageData(train_size=.75, gen_new_images=True)
    cn11 = ConvNet(cross_validating=True, model_num=209, conv_layers=(32, 64, 128, 128),
                    dense_layers=(512, 256), epochs=500, learning_rate=0.0001, dropout=0, patience=10)
    cn11.fit(dat1)
    sub11 = cn11.evaluate(dat1)


if __name__ == '__main__':
    main()
