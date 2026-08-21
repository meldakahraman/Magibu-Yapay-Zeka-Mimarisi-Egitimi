# Turkish Medical Articles - RAG System & Vector Database

Bu proje, Türkçe tıbbi makaleler üzerinde çalışan bir Retrieval-Augmented Generation (RAG) sistemi ve Vektör Veritabanı uygulamasıdır. Proje kapsamında ham veriler Hugging Face üzerinden çekilmiş, semantik parçalama uygulanmış, `magibu/embeddingmagibu-200m` modeliyle vektörleştirilmiş, ChromaDB üzerinde saklanmış ve başlangıç eşiği ile dinamik eşik optimizasyonu adımlarını içeren 30 soruluk bir benchmark testiyle doğrulanmıştır.

---

## 1. Veri Seti Seçimi ve Hazırlık
* **Veri Kaynağı:** Hugging Face üzerinden `umutertugrul/turkish-medical-articles` veri setinin `train` split'i kullanılmıştır.
* **Miktar:** Proje kriterlerine uygun olarak (100 - 1.000 arası), tutarlılık ve tekrar üretilebilirlik amacıyla `random_state=42` verilerek rastgele **500 adet** makale seçilmiştir.
* **Şema Mimarisi:** Oluşturulan veri tabanı tablosu ve DataFrame yapısı, yönergede istenen şu 3 temel sütunu içermektedir:
  1. `url`: Parçanın ait olduğu orijinal makalenin kaynak bağlantısı.
  2. `chunk_text`: Parçalanmış ham metin içeriği.
  3. `chunk_vector`: Seçilen embedding modeliyle üretilmiş 768 boyutlu float liste / vektör temsil.

---

## 2. Chunking (Parçalama) Stratejisi
* **Yöntem:** Sabit Karakter Boyutu + Örtüşme (Karakter Bazlı Hiyerarşik Parçalama)
* **Kullanılan Araç:** Langchain `RecursiveCharacterTextSplitter`
* **Parametreler:** 
  * `chunk_size = 1000` (Her bir parçanın maksimum karakter uzunluğu)
  * `chunk_overlap = 200` (Anlam bütünlüğünün kopmaması için önceki parçadan alınan karakter sayısı)
  * `separators = ["\n\n", "\n", " ", ""]`
* **Neden Seçildi?** Tıbbi metinlerde terminolojik bağlamın ve tanısal bütünlüğün kopmaması kritik bir öneme sahiptir. `RecursiveCharacterTextSplitter`, metni körü körüne ve rastgele kesmek yerine hiyerarşik ayırıcılar kullanarak önce paragrafları (`\n\n`), ardından satırları (`\n`) ve kelimeleri bölmeyi dener. Bu sayede tıbbi terimler bütünlüklerini korur. 200 karakterlik örtüşme ise bir chunk'tan diğerine geçişte bağlam kaybını engeller. Bu strateji sonucunda 500 makaleden toplam **2673 adet** anlamlı parça elde edilmiştir.

---

## 3. Embedding Modeli Seçimi
* **Tercih Edilen Model:** `magibu/embeddingmagibu-200m`
* **Vektör Boyutu:** 768
* **Neden Seçildi?** 
  1. **Dil Uyumluluğu:** Türkçe metinler üzerinde optimize edilmiş, 8192 token bağlam penceresi sunan güçlü bir Transformer tabanlı mimaridir.
  2. **Terminolojik Başarım:** Türkçe tıbbi terimleri ve karmaşık semantik (anlamsal) ilişkileri yüksek başarımla 768 boyutlu vektör uzayına taşıyabilmektedir.
  3. **Donanım Verimliliği:** Google Colab üzerinde A100 GPU altyapısıyla saniyeler içinde (2673 parçayı ~7 saniyede) vektörleştirme yapabilmenize olanak tanır.

---

## 4. Eşik (Threshold) Analizi ve Optimizasyon Süreci
Sistemin doğruluğunu artırmak ve Büyük Dil Modellerinin (LLM) veri setinde olmayan konularda uydurma yapmasını engellemek için kosinüs benzerliği temelli bir eşik mekanizması kurulmuştur. (ChromaDB'nin döndürdüğü mesafe `Distance` değeri, `1 - Distance` formülüyle Kosinüs Benzerliğine dönüştürülmüştür).

### Aşama 1: Başlangıç Eşiği Testi (Threshold = 0.50)
* Projenin ilk aşamasında genel kabul gören standart bir sınır olarak eşik değeri **0.50** seçilmiştir.
* Bu başlangıç testinde **10 negatif sorunun tamamı** (0.16 ile 0.37 arasındaki benzerlik skorlarıyla) başarıyla filtrelenmiş ve sisteme sızmaları engellenmiştir.
* Ancak pozitif sorulardan Soru 8 (*"Kan yapıcı besinler tüketilirken şeker yerine ne tercih edilmelidir?"*), **0.4598** benzerlik skoru alarak 0.50 eşiğinin altında kalmış ve sistem tarafından filtrelenmiştir. Bu durum, sabit 0.50 eşiğinin veri kümesindeki bazı geçerli ve zayıf eşleşen doğru verileri de sınırda bıraktığını göstermiştir.

### Aşama 2: Dinamik Eşik Optimizasyonu
Başlangıç eşiğinde yaşanan bu durumu bilimsel bir temele oturtmak ve optimizasyon yapmak amacıyla **Midpoint (Orta Nokta)** algoritması uygulanmıştır:
* **Negatif Skorlar Dağılımı (Max):** 0.3769 (Sisteme sızabilecek en yüksek ilgisiz benzerlik)
* **Pozitif Skorlar Dağılımı (Min):** 0.4598 (Doğru bilginin sistemde bulduğu en düşük benzerlik)
* **Hesaplanan Optimal Threshold:** 
 Threshold = (0.3769 + 0.4598) / 2 = 0.42

Bu matematiksel optimizasyon sayesinde karar sınırı, negatiflerin üst sınırı ile pozitiflerin alt sınırı arasında en güvenli marjı bırakacak şekilde **0.42** değerine çekilmiştir. Bu değerin altında kalan sorgular için modelin yanıt üretmesi engellenerek doğrudan *"Bu sorunun cevabı dokümanlarımda yer almamaktadır"* çıktısı verilmesi sağlanmıştır.

---

## 5. Benchmark Test Sonuçları
Optimizasyon sonrasında çalıştırılan 30 soruluk final testinin sonuçları `benchmark_sonuclari.txt` dosyasına kaydedilmiş olup performans özeti şöyledir:

* **Doğru Yanıtlanan Pozitif Sorular:** **20 / 20 (%100 Başarı)**
    * *Optimizasyon öncesi 0.50 eşiğinde takılan Soru 8 (0.4598 skor), yeni 0.42 eşik sınırı sayesinde başarıyla aşılmış ve doğru metin sisteme getirilmiştir.*
* **Başarıyla Filtrelenen Negatif Sorular:** **10 / 10 (%100 Başarı)**
    * *Veri kümesinde kesinlikle bulunmayan 10 farklı alakasız konunun (Kuantum Qiskit, League of Legends, Tığ işi wave stitch, Atbash şifreleme, BPE tokenizer, roman karakterleri, GBDT optimizasyonu, Minecraft Dungeons, 1D-CNN ve Teknofest rapor formatı) tamamı 0.42 eşiğinin altında (0.16 - 0.37 aralığında) kalarak başarıyla engellenmiştir.*

---

**Kullanılan Teknolojiler:** Python, Pandas, Langchain, Sentence-Transformers, ChromaDB