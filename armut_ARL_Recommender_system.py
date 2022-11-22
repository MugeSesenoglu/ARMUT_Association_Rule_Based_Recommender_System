###############################################################
# Association Rule Based Recommender System
###############################################################

#Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
#Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
#Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association
#Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

########################################
#  Veri Seti Hikayesi
########################################

#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# Değişkenler
#UserId : Müşteri numarası
#ServiceId : Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
#            Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
#            (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
#CategoryId : Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
#CreateDate : Hizmetin satın alındığı tarih

###############################################################
# PROJE Görevleri
###############################################################

##########################################################################################################
# GÖREV 1: Veriyi Hazırlama ve Analiz Etme
##########################################################################################################

#Adım 1: armut_data.csv dosyasını okutunuz.

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("datasets/armut_data.csv")
df.head()

#Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["ServiceId"] = df["ServiceId"].astype("str")
df["CategoryId"] = df["CategoryId"].astype("str")
df["Hizmet"] = ["_".join(col) for col in (df[["ServiceId","CategoryId"]].values)]
df.head()

#Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti; 2017'in 9.ayında aldığı 17_5, 14_7
#hizmetler başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["CreateDate"].head()

df["NewDate"] = df["CreateDate"].dt.strftime("%Y-%m")
df["NewDate"].head()

df["UserId"] = df["UserId"].astype("str")
df["NewDate"] = df["NewDate"].astype("str")
df["SepetId"] = ["_".join(col) for col in (df[["UserId","NewDate"]].values)]
df.head()

##################################################################################
# GÖREV 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
##################################################################################

#Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

df.groupby(["SepetId","New_Service"])["CategoryId"].count().unstack()
inv_pro_df =df.groupby(["SepetId","New_Service"])["CategoryId"]. \
    count().\
    unstack().\
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

inv_pro_df.shape

#Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.01) & (rules["confidence"]>0.1) & (rules["lift"]>1)]. \
sort_values("confidence", ascending=False)

#Adım3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0",5)
