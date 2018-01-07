from ImageData import ImageData
from ConvNet import ConvNet
import pandas as pd

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
                             sub2.is_iceberg, 
                             sub3.is_iceberg, 
                             sub4.is_iceberg, 
                             sub5.is_iceberg]).T

    ensemble['avg'] = ensemble.iloc[:, 1:].mean(axis=1)
    ensemble = ensemble[['id', 'avg']]
    ensemble.columns = ['id', 'is_iceberg']

    ensemble.to_csv('./ensemble5.csv', index=False)

if __name__ == '__main__':
    main()

    