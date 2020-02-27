from ap2p_model import Ap2p
from PIL import Image

インスタンス化

ap2p = Ap2p()


モデル確認用

ap2p.generator.summary()

ap2p.discriminator.summary()

ap2p.combined.summary()


重み読み込み

ap2p.load_weights( path )


学習

loss = ap2p.train(imgs_source, attr)

第一引数は　(128*128画素)の画像のリスト (Imageクラス）

第二引数は　画像リストに対応した属性情報　(numpy一次元配列, 0 or 1)

画像のリストの長さはミニバッチサイズ


重み保存

ap2p.save_weights( path )



予測 

img_attr, img_non_attr = ap2p.pred(imgs,5)

第一引数は　(128*128画素)の画像のリスト (Imageクラス）

第二引数は　属性情報の強度　(スカラー値, 目安：1 <= 強度 <=5)

img_attr 指定した強度で属性on

img_non_attr 属性off

img_attr, img_non_attr は　-1 から 1 の画素値