import pandas as pd
import numpy as np
import re
import datetime
import pickle
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Results:
    def __init__(self):
        self.results = pd.DataFrame()
        self.hr = pd.DataFrame()
        self.merged = pd.DataFrame()
    
    def p_p(self, results):
        self.results = results.copy()
        self.results.loc[self.results["course_len"]<1000, "course_len"] = 3600 # ステイヤーズS 中山２周となっているので修正いれておく
        self.results["着順"] = pd.to_numeric(self.results["着順"], errors="coerce")
        self.results["sex"] = self.results["性齢"].map(lambda x: str(x)[0])
        self.results["年齢"] = self.results["性齢"].map(lambda x: str(x)[1:]).astype(int)
        self.results["date"] = pd.to_datetime(self.results["date"], format="%Y年%m月%d日")
        self.results["month"] = self.results["date"].map(lambda x: x.month)
        month_dict = {3:"春", 4:"春", 5:"春", 6:"夏", 7:"夏", 8:"夏", 9:"秋", 10:"秋", 11:"秋", 12:"冬", 1:"冬", 2:"冬"}
        self.results["季節"] = self.results["month"].map(month_dict)
        self.results["race_id"] = self.results.index
        self.results = self.results.drop(columns={"馬名","性齢","タイム","着差","単勝","馬体重","調教師"})
    
    def merge(self, horse_results):
        self.hr = horse_results.copy()
        self.hr["date"] = pd.to_datetime(self.hr["日付"])
        self.hr.drop(['日付'], axis=1, inplace=True)
        self.hr["month"] = self.hr["date"].map(lambda x: x.month)
        month_dict = {3:"春", 4:"春", 5:"春", 6:"夏", 7:"夏", 8:"夏", 9:"秋", 10:"秋", 11:"秋", 12:"冬", 1:"冬", 2:"冬"}
        self.hr["季節"] = self.hr["month"].map(month_dict)
        self.hr["distance"] = self.hr["距離"].map(lambda x: str(x)[1:])
        self.hr["distance"] = self.hr["distance"].astype(int)
        self.hr["着順"] = pd.to_numeric(self.hr["着順"], errors="coerce")
        self.hr["実質着順"] = (1 - self.hr["着順"] / self.hr["頭数"])
        race_type_dict = {"ダ":"ダート", "障":"障害"}
        self.hr["race_type"] = self.hr["距離"].map(lambda x: str(x)[0])
        self.hr["race_type"] = self.hr["race_type"].map(race_type_dict)
        self.hr["馬体重"].str.split(r"\(", expand=True)
        self.hr["体重"] = self.hr["馬体重"].str.split(r"\(", expand=True)[0]
        self.hr["体重"] = pd.to_numeric(self.hr["体重"], errors="coerce")
        keibajo_dict = {"函館":"01", "札幌":"02", "福島":"03", "新潟":"04", "東京":"05", "中山":"06", "中京":"07", "京都":"08", "阪神":"09", "小倉":"10"}
        self.hr["競馬場"] = self.hr["開催"].map(lambda x: str(x)[1:3])
        self.hr["競馬場"] = self.hr["競馬場"].map(keibajo_dict)
        def corner(x, n):
            if type(x) != str:
                return x
            elif n==1:
                return int(re.findall(r'\d+', x)[0])
        self.hr["first_corner"] = self.hr["通過"].map(lambda x: corner(x, 1))
        
        date_list = self.results["date"].unique()
        merged_dict = {}
        for date in tqdm(date_list):
            filtered_hr = self.hr.query('date < @date')
            filtered_results = self.results.query('date == @date')
            merged_df = pd.merge(filtered_results, pd.DataFrame(filtered_hr.groupby("horse_id").size(), columns=["出走回数"]), on="horse_id", how="left")
            keibajo_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
            filtered_hr = filtered_hr.loc[filtered_hr.競馬場.isin(keibajo_list)]
            merged_df = pd.merge(merged_df, pd.DataFrame(filtered_hr.query('着順==1 or 着順==2')\
                                                         .groupby("horse_id")["季節"].sum()).rename(columns={"季節":"実績季節"}), on="horse_id", how="left")
            merged_df = pd.merge(merged_df, pd.DataFrame(filtered_hr.query('着順==1 or 着順==2')\
                                                         .groupby("horse_id")["jockey_id"].sum()).rename(columns={"jockey_id":"実績騎手"}), on="horse_id", how="left")
            merged_df = pd.merge(merged_df, filtered_hr.groupby("horse_id").nth(0)[["馬番","人気","体重","着順","着差","date","distance","first_corner","上り"]]\
                                 .add_suffix("_1走前"), on="horse_id", how="left")
            merged_df = pd.merge(merged_df, filtered_hr.groupby("horse_id").nth(1)[["実質着順","着順","着差","date","distance","上り"]].add_suffix("_2走前"),\
                     on="horse_id", how="left") 
            merged_df = pd.merge(merged_df, filtered_hr.groupby("horse_id").nth(2)[["実質着順","着差","date","distance","上り"]].add_suffix("_3走前"),\
                     on="horse_id", how="left") 
            merged_df["前々走距離変化"] = merged_df["distance_3走前"] - merged_df["distance_2走前"]
            merged_df["前走距離変化"] = merged_df["distance_2走前"] - merged_df["distance_1走前"]
            merged_df["今回距離変化"] = merged_df["distance_1走前"] - merged_df["course_len"]
            merged_df["前々走間隔"] = merged_df["date_2走前"] - merged_df["date_3走前"]
            merged_df["前走間隔"] = merged_df["date_1走前"] - merged_df["date_2走前"]
            merged_df["今回間隔"] = merged_df["date"] - merged_df["date_1走前"]
            merged_dict[date] = merged_df
        self.merged = pd.concat([merged_dict[key] for key in merged_dict])
        
        
class Peds:
    def __init__(self):
        self.df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        peds = pd.read_pickle('D:/jplb_workspace/peds_raw.pickle')
        peds_20 = pd.read_pickle('D:/jplb_workspace/peds_raw_20.pickle')
        self.peds = pd.concat([peds, peds_20])
        #self.peds = peds.copy()
        peds_copy = self.peds.copy()
        self.pointed = pd.DataFrame()
        
    def p_p_1(self, peds_copy):     
        
        peds_copy = peds_copy.rename(columns={0:"1代", 1:"2代", 2:"3代", 3:"4代", 4:"5代"})

        #小系統の分類をひとつずづ列作っていきます。実質的な小系統ワンホット。
        peds_copy["マイバブー系"] = ((peds_copy["1代"].str.contains("メジロマックイーン|トウカイテイオー")) | (peds_copy["2代"].str.contains("メジロマックイーン|トウカイテイオー")) | (peds_copy["3代"].str.contains("メジロマックイーン|トウカイテイオー")) | (peds_copy["4代"].str.contains("メジロマックイーン|トウカイテイオー")) | (peds_copy["5代"].str.contains("メジロマックイーン|トウカイテイオー"))).astype(int)
        peds_copy["マイバブー系"] = peds_copy["マイバブー系"].astype(str)
        peds_copy["ウォーニング系"] = ((peds_copy["1代"].str.contains("Warning")) | (peds_copy["2代"].str.contains("Warning")) | (peds_copy["3代"].str.contains("Warning")) | (peds_copy["4代"].str.contains("Warning")) | (peds_copy["5代"].str.contains("Warning"))).astype(int)
        peds_copy["ウォーニング系"] = peds_copy["ウォーニング系"].astype(str)
        peds_copy["リローンチ系"] = ((peds_copy["1代"].str.contains("Relaunch")) | (peds_copy["2代"].str.contains("Relaunch")) | (peds_copy["3代"].str.contains("Relaunch")) | (peds_copy["4代"].str.contains("Relaunch")) | (peds_copy["5代"].str.contains("Relaunch"))).astype(int)
        peds_copy["リローンチ系"] = peds_copy["リローンチ系"].astype(str)
        peds_copy["スターリング系"] = ((peds_copy["1代"].str.contains("Monsun")) | (peds_copy["2代"].str.contains("Monsun")) | (peds_copy["3代"].str.contains("Monsun")) | (peds_copy["4代"].str.contains("Monsun")) | (peds_copy["5代"].str.contains("Monsun"))).astype(int)
        peds_copy["スターリング系"] = peds_copy["スターリング系"].astype(str)
        peds_copy["米国マイナー系"] = ((peds_copy["1代"].str.contains("Damascus|Icecapade|Wild Again|Holy Bull|Broad Brush")) | (peds_copy["2代"].str.contains("Damascus|Icecapade|Wild Again|Holy Bull|Broad Brush")) | (peds_copy["3代"].str.contains("Damascus|Icecapade|Wild Again|Holy Bull|Broad Brush")) | (peds_copy["4代"].str.contains("Damascus|Icecapade|Wild Again|Holy Bull|Broad Brush")) | (peds_copy["5代"].str.contains("Damascus|Icecapade|Wild Again|Holy Bull|Broad Brush"))).astype(int)
        peds_copy["米国マイナー系"] = peds_copy["米国マイナー系"].astype(str)
        peds_copy["リボー系"] = ((peds_copy["1代"].str.contains("Graustark|His Majesty|Tom Rolfe|Ribot")) | (peds_copy["2代"].str.contains("Graustark|His Majesty|Tom Rolfe|Ribot")) | (peds_copy["3代"].str.contains("Graustark|His Majesty|Tom Rolfe|Ribot")) | (peds_copy["4代"].str.contains("Graustark|His Majesty|Tom Rolfe|Ribot")) | (peds_copy["5代"].str.contains("Graustark|His Majesty|Tom Rolfe|Ribot"))).astype(int)
        peds_copy["リボー系"] = peds_copy["リボー系"].astype(str)
        peds_copy["ハンプトン系"] = ((peds_copy["1代"].str.contains("Dictus|Surumu|Forli|Star Kingdom")) | (peds_copy["2代"].str.contains("Dictus|Surumu|Forli|Star Kingdom")) | (peds_copy["3代"].str.contains("Dictus|Surumu|Forli|Star Kingdom")) | (peds_copy["4代"].str.contains("Dictus|Surumu|Forli|Star Kingdom")) | (peds_copy["5代"].str.contains("Dictus|Surumu|Forli|Star Kingdom"))).astype(int)
        peds_copy["ハンプトン系"] = peds_copy["ハンプトン系"].astype(str)
        peds_copy["ニジンスキー系"] = ((peds_copy["1代"].str.contains("Nijinsky")) | (peds_copy["2代"].str.contains("Nijinsky")) | (peds_copy["3代"].str.contains("Nijinsky")) | (peds_copy["4代"].str.contains("Nijinsky")) | (peds_copy["5代"].str.contains("Nijinsky"))).astype(int)
        peds_copy["ニジンスキー系"] = peds_copy["ニジンスキー系"].astype(str)
        peds_copy["欧州ND系"] = ((peds_copy["1代"].str.contains("Lyphard|Dancing Brave|ホワイトマズル|キングヘイロー|Nureyev|Pivotal|Last Tycoon|Fairy King|Assatis")) | (peds_copy["2代"].str.contains("Lyphard|Dancing Brave|ホワイトマズル|キングヘイロー|Nureyev|Pivotal|Last Tycoon|Fairy King|Assatis")) | (peds_copy["3代"].str.contains("Lyphard|Dancing Brave|ホワイトマズル|キングヘイロー|Nureyev|Pivotal|Last Tycoon|Fairy King|Assatis")) | (peds_copy["4代"].str.contains("Lyphard|Dancing Brave|ホワイトマズル|キングヘイロー|Nureyev|Pivotal|Last Tycoon|Fairy King|Assatis")) | (peds_copy["5代"].str.contains("Lyphard|Dancing Brave|ホワイトマズル|キングヘイロー|Nureyev|Pivotal|Last Tycoon|Fairy King|Assatis"))).astype(int)
        peds_copy["欧州ND系"] = peds_copy["欧州ND系"].astype(str)
        peds_copy["サドラー系"] = ((peds_copy["1代"].str.contains("Sadler's Wells")) | (peds_copy["2代"].str.contains("Sadler's Wells")) | (peds_copy["3代"].str.contains("Sadler's Wells")) | (peds_copy["4代"].str.contains("Sadler's Wells")) | (peds_copy["5代"].str.contains("Sadler's Wells"))).astype(int)
        peds_copy["サドラー系"] = peds_copy["サドラー系"].astype(str)
        peds_copy["欧州ダンチヒ系"] = ((peds_copy["1代"].str.contains("Green Desert|Danehill")) | (peds_copy["2代"].str.contains("Green Desert|Danehill")) | (peds_copy["3代"].str.contains("Green Desert|Danehill")) | (peds_copy["4代"].str.contains("Green Desert|Danehill")) | (peds_copy["5代"].str.contains("Green Desert|Danehill"))).astype(int)
        peds_copy["欧州ダンチヒ系"] = peds_copy["欧州ダンチヒ系"].astype(str)
        peds_copy["米国ダンチヒ系"] = ((peds_copy["1代"].str.contains("Chief's Crown|Chief Bearhart|Hard Spun")) | (peds_copy["2代"].str.contains("Chief's Crown|Chief Bearhart|Hard Spun")) | (peds_copy["3代"].str.contains("Chief's Crown|Chief Bearhart|Hard Spun")) | (peds_copy["4代"].str.contains("Chief's Crown|Chief Bearhart|Hard Spun")) | (peds_copy["5代"].str.contains("Chief's Crown|Chief Bearhart|Hard Spun"))).astype(int)
        peds_copy["米国ダンチヒ系"] = peds_copy["米国ダンチヒ系"].astype(str)
        peds_copy["ヴァイスリージェント系"] = ((peds_copy["1代"].str.contains("Vice Regent|Deputy Minister")) | (peds_copy["2代"].str.contains("Vice Regent|Deputy Minister")) | (peds_copy["3代"].str.contains("Vice Regent|Deputy Minister")) | (peds_copy["4代"].str.contains("Vice Regent|Deputy Minister")) | (peds_copy["5代"].str.contains("Vice Regent|Deputy Minister"))).astype(int)
        peds_copy["ヴァイスリージェント系"] = peds_copy["ヴァイスリージェント系"].astype(str)
        peds_copy["ストームバード系"] = ((peds_copy["1代"].str.contains("Storm Bird|Storm Cat")) | (peds_copy["2代"].str.contains("Storm Bird|Storm Cat")) | (peds_copy["3代"].str.contains("Storm Bird|Storm Cat")) | (peds_copy["4代"].str.contains("Storm Bird|Storm Cat")) | (peds_copy["5代"].str.contains("Storm Bird|Storm Cat"))).astype(int)
        peds_copy["ストームバード系"] = peds_copy["ストームバード系"].astype(str)
        peds_copy["ノーザンテースト系"] = ((peds_copy["1代"].str.contains("ノーザンテースト")) | (peds_copy["2代"].str.contains("ノーザンテースト")) | (peds_copy["3代"].str.contains("ノーザンテースト")) | (peds_copy["4代"].str.contains("ノーザンテースト")) | (peds_copy["5代"].str.contains("ノーザンテースト"))).astype(int)
        peds_copy["ノーザンテースト系"] = peds_copy["ノーザンテースト系"].astype(str)
        peds_copy["ヘイロー系"] = ((peds_copy["1代"].str.contains("Devil|Glorious Song|サザンヘイロー")) | (peds_copy["2代"].str.contains("Devil|Glorious Song|サザンヘイロー")) | (peds_copy["3代"].str.contains("Devil|Glorious Song|サザンヘイロー")) | (peds_copy["4代"].str.contains("Devil|Glorious Song|サザンヘイロー")) | (peds_copy["5代"].str.contains("Devil|Glorious Song|サザンヘイロー"))).astype(int)
        peds_copy["ヘイロー系"] = peds_copy["ヘイロー系"].astype(str)
        peds_copy["ロベルト系"] = ((peds_copy["1代"].str.contains("Roberto")) | (peds_copy["2代"].str.contains("Roberto")) | (peds_copy["3代"].str.contains("Roberto")) | (peds_copy["4代"].str.contains("Roberto")) | (peds_copy["5代"].str.contains("Roberto"))).astype(int)
        peds_copy["ロベルト系"] = peds_copy["ロベルト系"].astype(str)
        peds_copy["サーゲイロード系"] = ((peds_copy["1代"].str.contains("Habitat|Sir Tristram")) | (peds_copy["2代"].str.contains("Habitat|Sir Tristram")) | (peds_copy["3代"].str.contains("Habitat|Sir Tristram")) | (peds_copy["4代"].str.contains("Habitat|Sir Tristram")) | (peds_copy["5代"].str.contains("Habitat|Sir Tristram"))).astype(int)
        peds_copy["サーゲイロード系"] = peds_copy["サーゲイロード系"].astype(str)
        peds_copy["米国ネイティヴ系"] = ((peds_copy["1代"].str.contains("Kauai King|Majestic Prince|Affirmed|Alydar")) | (peds_copy["2代"].str.contains("Kauai King|Majestic Prince|Affirmed|Alydar")) | (peds_copy["3代"].str.contains("Kauai King|Majestic Prince|Affirmed|Alydar")) | (peds_copy["4代"].str.contains("Kauai King|Majestic Prince|Affirmed|Alydar")) | (peds_copy["5代"].str.contains("Kauai King|Majestic Prince|Affirmed|Alydar"))).astype(int)
        peds_copy["米国ネイティヴ系"] = peds_copy["米国ネイティヴ系"].astype(str)
        peds_copy["欧州ネイティヴ系"] = ((peds_copy["1代"].str.contains("Atan|Sharpen Up|Sea-Bird")) | (peds_copy["2代"].str.contains("Atan|Sharpen Up|Sea-Bird")) | (peds_copy["3代"].str.contains("Atan|Sharpen Up|Sea-Bird")) | (peds_copy["4代"].str.contains("Atan|Sharpen Up|Sea-Bird")) | (peds_copy["5代"].str.contains("Atan|Sharpen Up|Sea-Bird"))).astype(int)
        peds_copy["欧州ネイティヴ系"] = peds_copy["欧州ネイティヴ系"].astype(str)
        peds_copy["キングマンボ系"] = ((peds_copy["1代"].str.contains("Kingmambo|Miswaki")) | (peds_copy["2代"].str.contains("Kingmambo")) | (peds_copy["3代"].str.contains("Kingmambo")) | (peds_copy["4代"].str.contains("Kingmambo")) | (peds_copy["5代"].str.contains("Kingmambo"))).astype(int)
        peds_copy["キングマンボ系"] = peds_copy["キングマンボ系"].astype(str)
        peds_copy["49er系"] = ((peds_copy["1代"].str.contains("サウスヴィグラス|プリサイスエンド|スウェプトオーヴァーボード|スイープトウショウ|ラインクラフト|アイルハヴアナザー|Coronado's Quest")) | (peds_copy["2代"].str.contains("サウスヴィグラス|プリサイスエンド|スウェプトオーヴァーボード|スイープトウショウ|ラインクラフト|アイルハヴアナザー|Coronado's Quest")) | (peds_copy["3代"].str.contains("サウスヴィグラス|プリサイスエンド|スウェプトオーヴァーボード|スイープトウショウ|ラインクラフト|アイルハヴアナザー|Coronado's Quest")) | (peds_copy["4代"].str.contains("サウスヴィグラス|プリサイスエンド|スウェプトオーヴァーボード|スイープトウショウ|ラインクラフト|アイルハヴアナザー|Coronado's Quest")) | (peds_copy["5代"].str.contains("サウスヴィグラス|プリサイスエンド|スウェプトオーヴァーボード|スイープトウショウ|ラインクラフト|アイルハヴアナザー|Coronado's Quest"))).astype(int)
        peds_copy["49er系"] = peds_copy["49er系"].astype(str)
        peds_copy["ダーレー系"] = ((peds_copy["1代"].str.contains("アドマイヤムーン")) | (peds_copy["2代"].str.contains("アドマイヤムーン")) | (peds_copy["3代"].str.contains("アドマイヤムーン")) | (peds_copy["4代"].str.contains("アドマイヤムーン")) | (peds_copy["5代"].str.contains("アドマイヤムーン"))).astype(int)
        peds_copy["ダーレー系"] = peds_copy["ダーレー系"].astype(str)
        peds_copy["ファピアノ系"] = ((peds_copy["1代"].str.contains("Fappiano")) | (peds_copy["2代"].str.contains("Fappiano")) | (peds_copy["3代"].str.contains("Fappiano")) | (peds_copy["4代"].str.contains("Fappiano")) | (peds_copy["5代"].str.contains("Fappiano"))).astype(int)
        peds_copy["ファピアノ系"] = peds_copy["ファピアノ系"].astype(str)
        peds_copy["その他ミスプロ系"] = ((peds_copy["1代"].str.contains("Woodman|Gone West|Seeking the Gold|Machiavellian|Smart Strike|King Glorious|アグネスデジタル|Afreet|Gulch|Jade Robbery|Scan|War Emblem|Aldebaran")) | (peds_copy["2代"].str.contains("Woodman|Gone West|Seeking the Gold|Machiavellian|Smart Strike|King Glorious|アグネスデジタル|Afreet|Gulch|Jade Robbery|Scan|War Emblem|Aldebaran")) | (peds_copy["3代"].str.contains("Woodman|Gone West|Seeking the Gold|Machiavellian|Smart Strike|King Glorious|アグネスデジタル|Afreet|Gulch|Jade Robbery|Scan|War Emblem|Aldebaran")) | (peds_copy["4代"].str.contains("Woodman|Gone West|Seeking the Gold|Machiavellian|Smart Strike|King Glorious|アグネスデジタル|Afreet|Gulch|Jade Robbery|Scan|War Emblem|Aldebaran")) | (peds_copy["5代"].str.contains("Woodman|Gone West|Seeking the Gold|Machiavellian|Smart Strike|King Glorious|アグネスデジタル|Afreet|Gulch|Jade Robbery|Scan|War Emblem|Aldebaran"))).astype(int)
        peds_copy["その他ミスプロ系"] = peds_copy["その他ミスプロ系"].astype(str)
        peds_copy["グレイソヴリン系"] = ((peds_copy["1代"].str.contains("シービークロス|Cozzene|Tony Bin|Caro")) | (peds_copy["2代"].str.contains("シービークロス|Cozzene|Tony Bin|Caro")) | (peds_copy["3代"].str.contains("シービークロス|Cozzene|Tony Bin|Caro")) | (peds_copy["4代"].str.contains("シービークロス|Cozzene|Tony Bin|Caro")) | (peds_copy["5代"].str.contains("シービークロス|Cozzene|Tony Bin|Caro"))).astype(int)
        peds_copy["グレイソヴリン系"] = peds_copy["グレイソヴリン系"].astype(str)
        peds_copy["プリンスリーギフト系"] = ((peds_copy["1代"].str.contains("サクラユタカオー")) | (peds_copy["2代"].str.contains("サクラユタカオー")) | (peds_copy["3代"].str.contains("サクラユタカオー")) | (peds_copy["4代"].str.contains("サクラユタカオー")) | (peds_copy["5代"].str.contains("サクラユタカオー"))).astype(int)
        peds_copy["プリンスリーギフト系"] = peds_copy["プリンスリーギフト系"].astype(str)
        peds_copy["ボールドルーラー系"] = ((peds_copy["1代"].str.contains("ロイヤルスキー|Seattle Slew")) | (peds_copy["2代"].str.contains("ロイヤルスキー|Seattle Slew")) | (peds_copy["3代"].str.contains("ロイヤルスキー|Seattle Slew")) | (peds_copy["4代"].str.contains("ロイヤルスキー|Seattle Slew")) | (peds_copy["5代"].str.contains("ロイヤルスキー|Seattle Slew"))).astype(int)
        peds_copy["ボールドルーラー系"] = peds_copy["ボールドルーラー系"].astype(str)
        peds_copy["レッドゴッド系"] = ((peds_copy["1代"].str.contains("Blushing Groom")) | (peds_copy["2代"].str.contains("Blushing Groom")) | (peds_copy["3代"].str.contains("Blushing Groom")) | (peds_copy["4代"].str.contains("Blushing Groom")) | (peds_copy["5代"].str.contains("Blushing Groom"))).astype(int)
        peds_copy["レッドゴッド系"] = peds_copy["レッドゴッド系"].astype(str)
        peds_copy["ネヴァーベンド系"] = ((peds_copy["1代"].str.contains("Shirley Heights|Mill Reef|ミルジョージ|Magnitude|Riverman|Bravest Roman")) | (peds_copy["2代"].str.contains("Shirley Heights|Mill Reef|ミルジョージ|Magnitude|Riverman|Bravest Roman")) | (peds_copy["3代"].str.contains("Shirley Heights|Mill Reef|ミルジョージ|Magnitude|Riverman|Bravest Roman")) | (peds_copy["4代"].str.contains("Shirley Heights|Mill Reef|ミルジョージ|Magnitude|Riverman|Bravest Roman")) | (peds_copy["5代"].str.contains("Shirley Heights|Mill Reef|ミルジョージ|Magnitude|Riverman|Bravest Roman"))).astype(int)
        peds_copy["ネヴァーベンド系"] = peds_copy["ネヴァーベンド系"].astype(str)
        peds_copy["ディープ系"] = ((peds_copy["1代"].str.contains("ディープインパクト")) | (peds_copy["2代"].str.contains("ディープインパクト")) | (peds_copy["3代"].str.contains("ディープインパクト")) | (peds_copy["4代"].str.contains("ディープインパクト")) | (peds_copy["5代"].str.contains("ディープインパクト"))).astype(int)
        peds_copy["ディープ系"] = peds_copy["ディープ系"].astype(str)
        peds_copy["Tサンデー系"] = ((peds_copy["1代"].str.contains("ブラックタイド|ステイゴールド|ゼンノロブロイ|ハーツクライ|ヴィクトワールピサ|マンハッタンカフェ|オルフェーヴル|ダンスインザダーク|スペシャルウィーク|アドマイヤグルーヴ")) | (peds_copy["2代"].str.contains("ブラックタイド|ステイゴールド|ゼンノロブロイ|ハーツクライ|ヴィクトワールピサ|マンハッタンカフェ|オルフェーヴル|ダンスインザダーク|スペシャルウィーク|アドマイヤグルーヴ")) | (peds_copy["3代"].str.contains("ブラックタイド|ステイゴールド|ゼンノロブロイ|ハーツクライ|ヴィクトワールピサ|マンハッタンカフェ|オルフェーヴル|ダンスインザダーク|スペシャルウィーク|アドマイヤグルーヴ")) | (peds_copy["4代"].str.contains("ブラックタイド|ステイゴールド|ゼンノロブロイ|ハーツクライ|ヴィクトワールピサ|マンハッタンカフェ|オルフェーヴル|ダンスインザダーク|スペシャルウィーク|アドマイヤグルーヴ")) | (peds_copy["5代"].str.contains("ブラックタイド|ステイゴールド|ゼンノロブロイ|ハーツクライ|ヴィクトワールピサ|マンハッタンカフェ|オルフェーヴル|ダンスインザダーク|スペシャルウィーク|アドマイヤグルーヴ"))).astype(int)
        peds_copy["Tサンデー系"] = peds_copy["Tサンデー系"].astype(str)
        peds_copy["Pサンデー系"] = ((peds_copy["1代"].str.contains("フジキセキ|ダイワメジャー|キンシャサノキセキ|デュランダル|マツリダゴッホ|ジョーカプチーノ|アグネスタキオン")) | (peds_copy["2代"].str.contains("フジキセキ|ダイワメジャー|キンシャサノキセキ|デュランダル|マツリダゴッホ|ジョーカプチーノ|アグネスタキオン")) | (peds_copy["3代"].str.contains("フジキセキ|ダイワメジャー|キンシャサノキセキ|デュランダル|マツリダゴッホ|ジョーカプチーノ|アグネスタキオン")) | (peds_copy["4代"].str.contains("フジキセキ|ダイワメジャー|キンシャサノキセキ|デュランダル|マツリダゴッホ|ジョーカプチーノ|アグネスタキオン")) | (peds_copy["5代"].str.contains("フジキセキ|ダイワメジャー|キンシャサノキセキ|デュランダル|マツリダゴッホ|ジョーカプチーノ"))).astype(int)
        peds_copy["Pサンデー系"] = peds_copy["Pサンデー系"].astype(str)
        peds_copy["Dサンデー系"] = ((peds_copy["1代"].str.contains("ゴールドアリュール|カネヒキリ|ネオユニヴァース|ディープスカイ|スズカマンボ")) | (peds_copy["2代"].str.contains("ゴールドアリュール|カネヒキリ|ネオユニヴァース|ディープスカイ|スズカマンボ")) | (peds_copy["3代"].str.contains("ゴールドアリュール|カネヒキリ|ネオユニヴァース|ディープスカイ|スズカマンボ")) | (peds_copy["4代"].str.contains("ゴールドアリュール|カネヒキリ|ネオユニヴァース|ディープスカイ|スズカマンボ")) | (peds_copy["5代"].str.contains("ゴールドアリュール|カネヒキリ|ネオユニヴァース|ディープスカイ|スズカマンボ"))).astype(int)
        peds_copy["Dサンデー系"] = peds_copy["Dサンデー系"].astype(str)
        #大系統と国系統型
        peds_copy["大系統サンデー"] = ((peds_copy["1代"].str.contains("サンデーサイレンス")) | (peds_copy["2代"].str.contains("サンデーサイレンス")) | (peds_copy["3代"].str.contains("サンデーサイレンス")) | (peds_copy["4代"].str.contains("サンデーサイレンス")) | (peds_copy["5代"].str.contains("サンデーサイレンス"))).astype(int)
        peds_copy["大系統サンデー"] = peds_copy["大系統サンデー"].astype(str)
        peds_copy["大系統ナスルーラ"] = ((peds_copy["グレイソヴリン系"]=="1") | (peds_copy["プリンスリーギフト系"]=="1") | (peds_copy["ボールドルーラー系"]=="1") | (peds_copy["レッドゴッド系"]=="1") | (peds_copy["ネヴァーベンド系"]=="1")).astype(int)
        peds_copy["大系統ナスルーラ"] = peds_copy["大系統ナスルーラ"].astype(str)
        peds_copy["大系統ミスプロ"] = ((peds_copy["キングマンボ系"]=="1") | (peds_copy["49er系"]=="1") | (peds_copy["ダーレー系"]=="1") | (peds_copy["ファピアノ系"]=="1") | (peds_copy["その他ミスプロ系"]=="1")).astype(int)
        peds_copy["大系統ミスプロ"] = peds_copy["大系統ミスプロ"].astype(str)
        peds_copy["大系統ターントゥ"] = ((peds_copy["ヘイロー系"]=="1") | (peds_copy["ロベルト系"]=="1") | (peds_copy["サーゲイロード系"]=="1")).astype(int)
        peds_copy["大系統ターントゥ"] = peds_copy["大系統ターントゥ"].astype(str)
        peds_copy["大系統米国ND"] = ((peds_copy["米国ダンチヒ系"]=="1") | (peds_copy["ヴァイスリージェント系"]=="1") | (peds_copy["ストームバード系"]=="1")).astype(int)
        peds_copy["大系統米国ND"] = peds_copy["大系統米国ND"].astype(str)
        peds_copy["大系統欧州ND"] = ((peds_copy["ニジンスキー系"]=="1") | (peds_copy["欧州ND系"]=="1") | (peds_copy["サドラー系"]=="1") | (peds_copy["欧州ダンチヒ系"]=="1")).astype(int)
        peds_copy["大系統欧州ND"] = peds_copy["大系統欧州ND"].astype(str)
        peds_copy["日本型"] = ((peds_copy["プリンスリーギフト系"]=="1") | (peds_copy["ダーレー系"]=="1") | (peds_copy["大系統サンデー"]=="1") | (peds_copy["ノーザンテースト系"]=="1")).astype(int)
        peds_copy["日本型"] = peds_copy["日本型"].astype(str)
        peds_copy["米国型"] = ((peds_copy["リローンチ系"]=="1") | (peds_copy["米国マイナー系"]=="1") | (peds_copy["大系統米国ND"]=="1") | (peds_copy["ヘイロー系"]=="1") | (peds_copy["米国ネイティヴ系"]=="1") | (peds_copy["49er系"]=="1") | (peds_copy["ファピアノ系"]=="1") | (peds_copy["その他ミスプロ系"]=="1") | (peds_copy["ボールドルーラー系"]=="1")).astype(int)
        peds_copy["米国型"] = peds_copy["米国型"].astype(str)
        peds_copy["欧州型"] = ((peds_copy["マイバブー系"]=="1") | (peds_copy["ウォーニング系"]=="1") | (peds_copy["スターリング系"]=="1") | (peds_copy["リボー系"]=="1") | (peds_copy["ハンプトン系"]=="1") | (peds_copy["大系統欧州ND"]=="1") | (peds_copy["ロベルト系"]=="1") | (peds_copy["サーゲイロード系"]=="1") | (peds_copy["欧州ネイティヴ系"]=="1") | (peds_copy["キングマンボ系"]=="1") | (peds_copy["グレイソヴリン系"]=="1") | (peds_copy["レッドゴッド系"]=="1") | (peds_copy["ネヴァーベンド系"]=="1")).astype(int)
        peds_copy["欧州型"] = peds_copy["欧州型"] .astype(str) 
        
        return peds_copy

    def p_p(self):
        peds_copy = self.peds.copy()
        
        titi = peds_copy[0::4]
        titi_p = self.p_p_1(titi)
        titi_p = titi_p.drop(columns={'2代', '3代', '4代', '5代', 'マイバブー系', 'ウォーニング系', 'リローンチ系', 'スターリング系',
               '米国マイナー系', 'リボー系', 'ハンプトン系', '欧州ND系', 'ヴァイスリージェント系', 'ストームバード系', 'ノーザンテースト系', 'ヘイロー系', 'ロベルト系',
               'サーゲイロード系', '米国ネイティヴ系', '欧州ネイティヴ系', '49er系', 'ダーレー系',
               'ファピアノ系', 'その他ミスプロ系', 'グレイソヴリン系', 'プリンスリーギフト系', 'ボールドルーラー系', 'レッドゴッド系',
               'ネヴァーベンド系'})
        titi_p.columns = '父_' + titi_p.columns
        hahatiti = peds_copy[2::4]
        hahatiti_p = self.p_p_1(hahatiti)
        hahatiti_p = hahatiti_p.drop(columns={'1代', '3代', '4代', '5代', 'マイバブー系', 'ウォーニング系', 'リローンチ系', 'スターリング系',
               '米国マイナー系', 'リボー系', 'ハンプトン系', '欧州ND系', 'ヴァイスリージェント系', 'ストームバード系', 'ノーザンテースト系', 'ヘイロー系', 'ロベルト系',
               'サーゲイロード系', '米国ネイティヴ系', '欧州ネイティヴ系', '49er系', 'ダーレー系',
               'ファピアノ系', 'その他ミスプロ系', 'グレイソヴリン系', 'プリンスリーギフト系', 'ボールドルーラー系', 'レッドゴッド系',
               'ネヴァーベンド系'})
        hahatiti_p.columns = '母父_' + hahatiti_p.columns
        hahahahatiti = peds_copy[3::4]
        hahahahatiti_p = self.p_p_1(hahahahatiti)
        hahahahatiti_p = hahahahatiti_p.drop(columns={'1代', '2代', '4代', '5代', 'マイバブー系', 'ウォーニング系', 'リローンチ系', 'スターリング系',
               '米国マイナー系', 'リボー系', 'ハンプトン系', 'ニジンスキー系', '欧州ND系', 'サドラー系', '欧州ダンチヒ系',
               '米国ダンチヒ系', 'ヴァイスリージェント系', 'ストームバード系', 'ノーザンテースト系', 'ヘイロー系', 'ロベルト系',
               'サーゲイロード系', '米国ネイティヴ系', '欧州ネイティヴ系', 'キングマンボ系', '49er系', 'ダーレー系',
               'ファピアノ系', 'その他ミスプロ系', 'グレイソヴリン系', 'プリンスリーギフト系', 'ボールドルーラー系', 'レッドゴッド系',
               'ネヴァーベンド系', 'ディープ系', 'Tサンデー系', 'Pサンデー系', 'Dサンデー系', 
               '大系統ナスルーラ', '大系統ミスプロ', '大系統ターントゥ', '大系統米国ND', '大系統欧州ND'})
        hahahahatiti_p.columns = '母母父_' + hahahahatiti_p.columns
        titihahatiti = peds_copy[1::4]
        titihahatiti_p = self.p_p_1(titihahatiti)
        titihahatiti_p = titihahatiti_p.drop(columns={'1代', '2代', '3代', '4代', '5代', 'マイバブー系', 'ウォーニング系', 'リローンチ系', 'スターリング系',
               '米国マイナー系', 'リボー系', 'ハンプトン系', 'ニジンスキー系', '欧州ND系', 'サドラー系', '欧州ダンチヒ系',
               '米国ダンチヒ系', 'ヴァイスリージェント系', 'ストームバード系', 'ノーザンテースト系', 'ヘイロー系', 'ロベルト系',
               'サーゲイロード系', '米国ネイティヴ系', '欧州ネイティヴ系', 'キングマンボ系', '49er系', 'ダーレー系',
               'ファピアノ系', 'その他ミスプロ系', 'グレイソヴリン系', 'プリンスリーギフト系', 'ボールドルーラー系', 'レッドゴッド系',
               'ネヴァーベンド系', 'ディープ系', 'Tサンデー系', 'Pサンデー系', 'Dサンデー系',
               '大系統ナスルーラ', '大系統ミスプロ', '大系統ターントゥ', '大系統米国ND', '大系統欧州ND', '日本型', '米国型',
               '欧州型'})
        titihahatiti_p.columns = '父母父_' + titihahatiti_p.columns

        peds_shinba = pd.concat([titi_p, hahatiti_p, titihahatiti_p, hahahahatiti_p], axis=1)  # return peds_shinba

        peds_shinba = peds_shinba.rename(columns={'父_1代':"父", '母父_2代':"母父", "母母父_3代":"母母父"})
        peds_shinba["父"] = peds_shinba["父"].str.split("\d+", expand=True)[0]
        peds_shinba["母父"] = peds_shinba["母父"].str.split("\d+", expand=True)[0]
        peds_shinba["母母父"] = peds_shinba["母母父"].str.split("\d+", expand=True)[0]
        peds_shinba["非サンデー馬"] = ((peds_shinba['父_大系統サンデー']=="0") & (peds_shinba['母父_大系統サンデー']=="0") & (peds_shinba['父母父_大系統サンデー']=="0") &\
                                 (peds_shinba['母母父_大系統サンデー']=="0")).astype(int)
        peds_shinba["非サンデー馬"] = peds_shinba["非サンデー馬"].astype(str)
        peds_shinba["サンデー_米国"] = (((peds_shinba["父_大系統サンデー"]=="1") & (peds_shinba["母父_米国型"]=="1")) | ((peds_shinba["母父_大系統サンデー"]=="1") &\
                                                                                                      (peds_shinba["父_米国型"]=="1"))).astype(int)
        peds_shinba["サンデー_米国"] = peds_shinba["サンデー_米国"].astype(str)
        peds_shinba["米国B"] = (((peds_shinba["父_米国型"]=="1") & (peds_shinba["母父_米国型"]=="1")) | ((peds_shinba["父_米国型"]=="1") & (peds_shinba["母母父_米国型"]=="1"))).astype(int)
        peds_shinba["米国B"] = peds_shinba["米国B"].astype(str)
        peds_shinba["米国A"] = ((peds_shinba["父_米国型"]=="1") & (peds_shinba["母父_米国型"]=="1") & (peds_shinba["母母父_米国型"]=="1")).astype(int)
        peds_shinba["米国A"] = peds_shinba["米国A"].astype(str)
        peds_shinba["父_母父_ディープ"] = ((peds_shinba["父_ディープ系"]=="1") | (peds_shinba["母父_ディープ系"]=="1")).astype(int)
        peds_shinba["父_母父_ディープ"] = peds_shinba["父_母父_ディープ"].astype(str)
        peds_shinba["父_母父_キングマンボ系"] = ((peds_shinba["父_キングマンボ系"]=="1") | (peds_shinba["母父_キングマンボ系"]=="1")).astype(int)
        peds_shinba["父_母父_キングマンボ系"] = peds_shinba["父_母父_キングマンボ系"].astype(str)
        peds_shinba["欧州A"] = ((peds_shinba["父_欧州型"]=="1") & (peds_shinba["母父_欧州型"]=="1") & (peds_shinba["母母父_欧州型"]=="1")).astype(int)
        peds_shinba["欧州A"] = peds_shinba["欧州A"].astype(str)
        peds_shinba["欧州B"] = (((peds_shinba["母父_欧州型"]=="1") & (peds_shinba["母母父_欧州型"]=="1")) | \
                              ((peds_shinba["父_欧州型"]=="1") & (peds_shinba["母母父_欧州型"]=="1")) | \
                             ((peds_shinba["父_欧州型"]=="1") & (peds_shinba["母父_欧州型"]=="1"))).astype(int)
        peds_shinba["欧州B"] = peds_shinba["欧州B"] .astype(str)
        peds_shinba["horse_id"] = peds_shinba.index
        
        self.df = peds_shinba.copy()
    
    def merge(self, results):
        df = pd.merge(results, self.df, on="horse_id", how="left")
        df["course_len"] = df["course_len"].astype(str)
        df["course"] = df["競馬場"]+df["race_type"]+df["course_len"]
        df["distance"] = df["course_len"].astype(int)
        df["blood_point"] = 0
        turf_a_list = ["06芝1600", "05芝1600", "08芝1600", "05芝1400", "09芝1600", "09芝1800", "05芝1800", "06芝1200", "08芝1800", "03芝1200",\
                       "09芝2000", "06芝2000", "06芝1800", "09芝1400", "08芝1400", "05芝2000", "04芝1000", "07芝1600", "09芝1200", "10芝1200",\
                       "08芝1200", "07芝1200"]
        turf_b_list = ["10芝1200", "03芝1200", "02芝1200", "10芝1800", "10芝2000", "08芝2000", "07芝2000", "01芝1200", "06芝2000", "08芝1600", "03芝2000",\
                       "03芝1800", "04芝1000", "09芝2000", "06芝1600", "09芝1400", "06芝1800", "09芝1600", "09芝1800", "06芝1200"]
        turf_c_list = ["10芝2000", "08芝2000", "03芝2000", "05芝2400", "01芝2000", "10芝1800", "03芝2600", "10芝2600", "06芝2000", "09芝2000", "09芝2400",\
                       "02芝2000", "07芝2200", "04芝1800", "04芝2000", "02芝1800", "08芝1800", "03芝1800", "06芝2200", "05芝2000", "07芝2000", "04芝2400",\
                       "08芝2400", "04芝2200", "02芝2600", "01芝2600", "08芝2200", "09芝2200"]
        turf_d_list = ["05芝1600", "05芝1800", "06芝1600", "05芝1400", "10芝1200", "05芝2000", "09芝1600", "06芝2000", "05芝2400", "03芝1200", "07芝2000",\
                       "04芝1800", "09芝1800", "08芝1600", "04芝2000", "08芝1800", "07芝1400", "04芝1600", "08芝1400", "02芝1200"]
        dart_a_list = ["06ダート1200", "08ダート1200", "09ダート1400", "09ダート1200", "04ダート1200", "05ダート1600", "08ダート1800", "05ダート1400",\
                       "08ダート1400", "09ダート1800", "06ダート1800", "07ダート1400", "07ダート1200", "03ダート1150", "10ダート1000", "10ダート1700",\
                       "05ダート1300", "01ダート1000"]
        dart_b_list = ["06ダート1800", "09ダート1800", "08ダート1800", "04ダート1800", "03ダート1700", "05ダート1600", "06ダート1200", "07ダート1800",\
                       "10ダート1700", "02ダート1700", "01ダート1700", "09ダート1400", "04ダート1200", "05ダート1400", "08ダート1400", "03ダート1150"]
        dart_c_list = ["06ダート1800", "09ダート1800", "04ダート1800", "05ダート1600", "08ダート1800", "05ダート2100", "03ダート1700", "07ダート1800",\
                       "10ダート1700", "02ダート1700", "01ダート1700", "05ダート1400", "06ダート1200", "09ダート1400", "06ダート2400", "07ダート1900",\
                       "09ダート2000"]
        dart_d_list = ["05ダート1400", "05ダート1600", "09ダート1800", "06ダート1200", "06ダート1800", "08ダート1800", "09ダート1400",\
                       "08ダート1400", "05ダート2100", "07ダート1800", "07ダート1400", "04ダート1800", "08ダート1200", "10ダート1700", "08ダート1900"]
        
        df.loc[(df.父=="ディープインパクト ") & ((df.course=="05芝1600") | (df.course=="05芝1800") | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] = \
            df.loc[(df.父=="ディープインパクト ") & ((df.course=="05芝1600") | (df.course=="05芝1800") | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] +1
        df.loc[(df.父=="ディープインパクト ") & (df.course.isin(turf_a_list+turf_c_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="ディープインパクト ") & (df.course.isin(turf_a_list+turf_c_list+turf_d_list)), "blood_point"] +1
        df.loc[(df.父=="ディープインパクト ") & ((df.course.isin(turf_b_list)) | ((df.sex=="牝") & (df.course.isin(turf_c_list)))), "blood_point"] = \
            df.loc[(df.父=="ディープインパクト ") & ((df.course.isin(turf_b_list)) | ((df.sex=="牝") & (df.course.isin(turf_c_list)))), "blood_point"] -1
        df.loc[(df.父=="ハーツクライ ") & (((df.sex=="牡") & (df.course.isin(turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_d_list)))), "blood_point"] = \
            df.loc[(df.父=="ハーツクライ ") & (((df.sex=="牡") & (df.course.isin(turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_d_list)))), "blood_point"] +1
        df.loc[(df.父=="ハーツクライ ") & (df.course.isin(turf_a_list+turf_c_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="ハーツクライ ") & (df.course.isin(turf_a_list+turf_c_list+turf_d_list)), "blood_point"] +1
        df.loc[(df.父=="ハーツクライ ") & ((df.sex=="牝") & (df.course.isin(turf_b_list))), "blood_point"] = \
            df.loc[(df.父=="ハーツクライ ") & ((df.sex=="牝") & (df.course.isin(turf_b_list))), "blood_point"] -1
        df.loc[(df.父=="ハーツクライ ") & (df.sex=="牡") & (df.course.isin(dart_b_list+dart_c_list+dart_d_list)), "blood_point"] = \
            df.loc[(df.父=="ハーツクライ ") & (df.sex=="牡") & (df.course.isin(dart_b_list+dart_c_list+dart_d_list)), "blood_point"] +1
        df.loc[(df.父=="ダイワメジャー ") & (df.course.isin(turf_a_list)), "blood_point"] = \
            df.loc[(df.父=="ダイワメジャー ") & (df.course.isin(turf_a_list)), "blood_point"] +2
        df.loc[(df.父=="ダイワメジャー ") & ((df.sex=="牡") & (df.course.isin(turf_d_list))), "blood_point"] = \
            df.loc[(df.父=="ダイワメジャー ") & ((df.sex=="牡") & (df.course.isin(turf_d_list))), "blood_point"] +1
        df.loc[(df.父=="ダイワメジャー ") & (df.course.isin(turf_b_list+turf_c_list)), "blood_point"] = \
            df.loc[(df.父=="ダイワメジャー ") & (df.course.isin(turf_b_list+turf_c_list)), "blood_point"] -1
        df.loc[(df.父=="ダイワメジャー ") & ((df.sex=="牡") & (df.course.isin(dart_a_list))), "blood_point"] = \
            df.loc[(df.父=="ダイワメジャー ") & ((df.sex=="牡") & (df.course.isin(dart_a_list))), "blood_point"] +1
        df.loc[(df.父=="ハービンジャー Harbinger(英) ") & (df.course.isin(turf_c_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="ハービンジャー Harbinger(英) ") & (df.course.isin(turf_c_list+turf_d_list)), "blood_point"] +2
        df.loc[(df.父=="ハービンジャー Harbinger(英) ") & ((df.sex=="牡") & (df.course.isin(turf_b_list))), "blood_point"] = \
            df.loc[(df.父=="ハービンジャー Harbinger(英) ") & ((df.sex=="牡") & (df.course.isin(turf_b_list))), "blood_point"] +1
        df.loc[(df.父=="ハービンジャー Harbinger(英) ") & (df.course.isin(turf_a_list)), "blood_point"] = \
            df.loc[(df.父=="ハービンジャー Harbinger(英) ") & (df.course.isin(turf_a_list)), "blood_point"] -1
        df.loc[(df.父=="ルーラーシップ ") & (((df.sex=="牡") & (df.course.isin(turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] = \
            df.loc[(df.父=="ルーラーシップ ") & (((df.sex=="牡") & (df.course.isin(turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] +2
        df.loc[(df.父=="ルーラーシップ ") & (((df.sex=="牡") & (df.course.isin(turf_b_list))) | ((df.sex=="牝") & (df.course.isin(turf_c_list)))), "blood_point"] = \
            df.loc[(df.父=="ルーラーシップ ") & (((df.sex=="牡") & (df.course.isin(turf_b_list))) | ((df.sex=="牝") & (df.course.isin(turf_c_list)))), "blood_point"] +1
        df.loc[(df.父=="ルーラーシップ ") & (df.course.isin(turf_a_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="ルーラーシップ ") & (df.course.isin(turf_a_list+turf_d_list)), "blood_point"] -1
        df.loc[(df.父=="ルーラーシップ ") & ((df.sex=="牡") & (df.course.isin(dart_c_list))), "blood_point"] = \
            df.loc[(df.父=="ルーラーシップ ") & ((df.sex=="牡") & (df.course.isin(dart_c_list))), "blood_point"] +1
        df.loc[(df.父=="ロードカナロア ") & (df.course.isin(turf_a_list)), "blood_point"] = \
            df.loc[(df.父=="ロードカナロア ") & (df.course.isin(turf_a_list)), "blood_point"] +2
        df.loc[(df.父=="ロードカナロア ") & ((df.sex=="牡") & (df.course.isin(turf_b_list))), "blood_point"] = \
            df.loc[(df.父=="ロードカナロア ") & ((df.sex=="牡") & (df.course.isin(turf_b_list))), "blood_point"] +1
        df.loc[(df.父=="ロードカナロア ") & (df.course.isin(turf_c_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="ロードカナロア ") & (df.course.isin(turf_c_list+turf_d_list)), "blood_point"] -1
        df.loc[(df.父=="ロードカナロア ") & (df.course.isin(dart_a_list+dart_b_list)), "blood_point"] = \
            df.loc[(df.父=="ロードカナロア ") & (df.course.isin(dart_a_list+dart_b_list)), "blood_point"] +1
        df.loc[(df.父=="ステイゴールド ") & ((df.sex=="牡") & (df.course.isin(turf_a_list+turf_d_list))), "blood_point"] = \
            df.loc[(df.父=="ステイゴールド ") & ((df.sex=="牡") & (df.course.isin(turf_a_list+turf_d_list))), "blood_point"] +2
        df.loc[(df.父=="ステイゴールド ") & (((df.sex=="牝") & (df.course.isin(turf_d_list))) | (df.course.isin(turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="ステイゴールド ") & (((df.sex=="牝") & (df.course.isin(turf_d_list))) | (df.course.isin(turf_c_list))), "blood_point"] +1
        df.loc[(df.父=="ステイゴールド ") & (df.course.isin(turf_b_list)), "blood_point"] = \
            df.loc[(df.父=="ステイゴールド ") & (df.course.isin(turf_b_list)), "blood_point"] -1
        df.loc[(df.父=="キングカメハメハ ") & (((df.sex=="牡") & (df.course.isin(turf_a_list))) | ((df.sex=="牝") & (df.course.isin(turf_d_list)))), "blood_point"] = \
            df.loc[(df.父=="キングカメハメハ ") & (((df.sex=="牡") & (df.course.isin(turf_a_list))) | ((df.sex=="牝") & (df.course.isin(turf_d_list)))), "blood_point"] +2
        df.loc[(df.父=="キングカメハメハ ") & (((df.sex=="牡") & (df.course.isin(turf_d_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] = \
            df.loc[(df.父=="キングカメハメハ ") & (((df.sex=="牡") & (df.course.isin(turf_d_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] +1
        df.loc[(df.父=="キングカメハメハ ") & (df.course.isin(turf_b_list)), "blood_point"] = \
            df.loc[(df.父=="キングカメハメハ ") & (df.course.isin(turf_b_list)), "blood_point"] -1
        df.loc[(df.父=="キングカメハメハ ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="キングカメハメハ ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] +1
        df.loc[(df.父=="オルフェーヴル ") & ((df.sex=="牡") & (df.course.isin(turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="オルフェーヴル ") & ((df.sex=="牡") & (df.course.isin(turf_c_list))), "blood_point"] +2
        df.loc[(df.父=="オルフェーヴル ") & ((df.sex=="牝") & (df.course.isin(turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="オルフェーヴル ") & ((df.sex=="牝") & (df.course.isin(turf_c_list))), "blood_point"] +1
        df.loc[(df.父=="オルフェーヴル ") & (df.course.isin(turf_a_list)), "blood_point"] = \
            df.loc[(df.父=="オルフェーヴル ") & (df.course.isin(turf_a_list)), "blood_point"] -1
        df.loc[(df.父=="オルフェーヴル ") & (df.course.isin(dart_c_list)), "blood_point"] = \
            df.loc[(df.父=="オルフェーヴル ") & (df.course.isin(dart_c_list)), "blood_point"] +1
        df.loc[(df.父=="ヴィクトワールピサ ") & ((df.sex=="牝") & (df.course.isin(turf_c_list+turf_d_list))), "blood_point"] = \
            df.loc[(df.父=="ヴィクトワールピサ ") & ((df.sex=="牝") & (df.course.isin(turf_c_list+turf_d_list))), "blood_point"] +1
        df.loc[(df.父=="キンシャサノキセキ ") & ((df.sex=="牡") & (df.course.isin(turf_a_list))), "blood_point"] = \
            df.loc[(df.父=="キンシャサノキセキ ") & ((df.sex=="牡") & (df.course.isin(turf_a_list))), "blood_point"] +2
        df.loc[(df.父=="キンシャサノキセキ ") & (((df.sex=="牡") & (df.course.isin(turf_b_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list+turf_d_list)))), "blood_point"] = \
            df.loc[(df.父=="キンシャサノキセキ ") & (((df.sex=="牡") & (df.course.isin(turf_b_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list+turf_d_list)))), "blood_point"] +1
        df.loc[(df.父=="キンシャサノキセキ ") & ((df.sex=="牡") & (df.course.isin(dart_a_list+dart_d_list))), "blood_point"] = \
            df.loc[(df.父=="キンシャサノキセキ ") & ((df.sex=="牡") & (df.course.isin(dart_a_list+dart_d_list))), "blood_point"] +1
        df.loc[(df.父=="マンハッタンカフェ ") & (df.course.isin(turf_a_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="マンハッタンカフェ ") & (df.course.isin(turf_a_list+turf_d_list)), "blood_point"] +1
        df.loc[(df.父=="ディープブリランテ ") & (((df.sex=="牡") & (df.course.isin(turf_d_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] = \
            df.loc[(df.父=="ディープブリランテ ") & (((df.sex=="牡") & (df.course.isin(turf_d_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] +1
        df.loc[(df.父=="スクリーンヒーロー ") & (((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] = \
            df.loc[(df.父=="スクリーンヒーロー ") & (((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] +1
        df.loc[(df.父=="ブラックタイド ") & (((df.sex=="牡") & (df.course.isin(turf_b_list))) | ((df.sex=="牝") & (df.course.isin(turf_d_list)))), "blood_point"] = \
            df.loc[(df.父=="ブラックタイド ") & (((df.sex=="牡") & (df.course.isin(turf_b_list))) | ((df.sex=="牝") & (df.course.isin(turf_d_list)))), "blood_point"] +1
        df.loc[(df.父=="ノヴェリスト Novellist(愛) ") & (((df.sex=="牡") & (df.course.isin(turf_a_list+turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] = \
            df.loc[(df.父=="ノヴェリスト Novellist(愛) ") & (((df.sex=="牡") & (df.course.isin(turf_a_list+turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] +1
        df.loc[(df.父=="エイシンフラッシュ ") & ((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="エイシンフラッシュ ") & ((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))), "blood_point"] +1
        df.loc[(df.父=="キズナ ") & ((df.sex=="牝") & (df.course.isin(turf_b_list))), "blood_point"] = \
            df.loc[(df.父=="キズナ ") & ((df.sex=="牝") & (df.course.isin(turf_b_list))), "blood_point"] +2
        df.loc[(df.父=="キズナ ") & (df.sex=="牡") & (((df.course.isin(turf_b_list))) | (df.race_type=="ダート")), "blood_point"] = \
            df.loc[(df.父=="キズナ ") & (df.sex=="牡") & (((df.course.isin(turf_b_list))) | (df.race_type=="ダート")), "blood_point"] +1
        df.loc[(df.父=="アドマイヤムーン ") & ((df.sex=="牡") & (df.course.isin(turf_a_list+turf_d_list))), "blood_point"] = \
            df.loc[(df.父=="アドマイヤムーン ") & ((df.sex=="牡") & (df.course.isin(turf_a_list+turf_d_list))), "blood_point"] +1
        df.loc[(df.父=="ジャスタウェイ ") & ((df.sex=="牡") & (df.course.isin(turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="ジャスタウェイ ") & ((df.sex=="牡") & (df.course.isin(turf_c_list))), "blood_point"] +1
        df.loc[(df.父=="エピファネイア ") & ((df.sex=="牝") & (df.course.isin(turf_b_list+turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="エピファネイア ") & ((df.sex=="牝") & (df.course.isin(turf_b_list+turf_c_list))), "blood_point"] +2
        df.loc[(df.父=="エピファネイア ") & ((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))), "blood_point"] = \
            df.loc[(df.父=="エピファネイア ") & ((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))), "blood_point"] +1
        df.loc[(df.父=="ドリームジャーニー ") & ((df.sex=="牡") & (df.course.isin(turf_c_list+turf_d_list))), "blood_point"] = \
            df.loc[(df.父=="ドリームジャーニー ") & ((df.sex=="牡") & (df.course.isin(turf_c_list+turf_d_list))), "blood_point"] +1
        df.loc[(df.父=="ジャングルポケット ") & ((df.sex=="牡") & (df.course.isin(turf_a_list+turf_d_list))), "blood_point"] = \
            df.loc[(df.父=="ジャングルポケット ") & ((df.sex=="牡") & (df.course.isin(turf_a_list+turf_d_list))), "blood_point"] +1
        df.loc[(df.父=="メイショウサムソン ") & (df.course.isin(turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="メイショウサムソン ") & (df.course.isin(turf_d_list)), "blood_point"] +1
        df.loc[(df.父=="ワークフォース Workforce(英) ") & (df.course.isin(turf_c_list)), "blood_point"] = \
            df.loc[(df.父=="ワークフォース Workforce(英) ") & (df.course.isin(turf_c_list)), "blood_point"] +1
        df.loc[(df.父=="マツリダゴッホ ") & ((df.sex=="牝") & (df.course.isin(turf_a_list))), "blood_point"] = \
            df.loc[(df.父=="マツリダゴッホ ") & ((df.sex=="牝") & (df.course.isin(turf_a_list))), "blood_point"] +2
        df.loc[(df.父=="マツリダゴッホ ") & (((df.sex=="牡") & (df.course.isin(turf_b_list+turf_d_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] = \
            df.loc[(df.父=="マツリダゴッホ ") & (((df.sex=="牡") & (df.course.isin(turf_b_list+turf_d_list))) | ((df.sex=="牝") & (df.course.isin(turf_b_list)))), "blood_point"] +1
        df.loc[(df.父=="ゴールドアリュール ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="ゴールドアリュール ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] +2
        df.loc[(df.父=="ゴールドアリュール ") & (df.sex=="牝") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="ゴールドアリュール ") & (df.sex=="牝") & (df.race_type=="ダート"), "blood_point"] +1
        df.loc[(df.父=="クロフネ ") & (df.sex=="牝") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="クロフネ ") & (df.sex=="牝") & (df.race_type=="ダート"), "blood_point"] +2
        df.loc[(df.父=="クロフネ ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="クロフネ ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] +1
        df.loc[(df.父=="サウスヴィグラス ") & (df.course.isin(dart_a_list)), "blood_point"] = \
            df.loc[(df.父=="サウスヴィグラス ") & (df.course.isin(dart_a_list)), "blood_point"] +2
        df.loc[(df.父=="ヘニーヒューズ Henny Hughes(米) ") & (df.course.isin(dart_a_list+dart_d_list)), "blood_point"] = \
            df.loc[(df.父=="ヘニーヒューズ Henny Hughes(米) ") & (df.course.isin(dart_a_list+dart_d_list)), "blood_point"] +2
        df.loc[(df.父=="エンパイアメーカー Empire Maker(米) ") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="エンパイアメーカー Empire Maker(米) ") & (df.race_type=="ダート"), "blood_point"] +1
        df.loc[(df.父=="アイルハヴアナザー I\'ll Have Another(米) ") & (df.course.isin(dart_c_list)), "blood_point"] = \
            df.loc[(df.父=="アイルハヴアナザー I\'ll Have Another(米) ") & (df.course.isin(dart_c_list)), "blood_point"] +1
        df.loc[(df.父=="シニスターミニスター Sinister Minister(米) ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="シニスターミニスター Sinister Minister(米) ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] +1
        df.loc[(df.父=="ネオユニヴァース ") & (df.sex=="牡") & (df.course.isin(dart_c_list+dart_d_list)), "blood_point"] = \
            df.loc[(df.父=="ネオユニヴァース ") & (df.sex=="牡") & (df.course.isin(dart_c_list+dart_d_list)), "blood_point"] +1
        df.loc[(df.父=="シンボリクリスエス ") & (df.sex=="牡") & (df.course.isin(dart_c_list+dart_d_list)), "blood_point"] = \
            df.loc[(df.父=="シンボリクリスエス ") & (df.sex=="牡") & (df.course.isin(dart_c_list+dart_d_list)), "blood_point"] +1
        df.loc[(df.父=="メイショウボーラー ") & (df.sex=="牡") & (df.course.isin(dart_a_list)), "blood_point"] = \
            df.loc[(df.父=="メイショウボーラー ") & (df.sex=="牡") & (df.course.isin(dart_a_list)), "blood_point"] +1
        df.loc[(df.父=="カネヒキリ ") & (df.course.isin(dart_a_list+dart_d_list)), "blood_point"] = \
            df.loc[(df.父=="カネヒキリ ") & (df.course.isin(dart_a_list+dart_d_list)), "blood_point"] +1
        df.loc[(df.父=="パイロ Pyro(米) ") & (df.course.isin(dart_a_list+dart_d_list)), "blood_point"] = \
            df.loc[(df.父=="パイロ Pyro(米) ") & (df.course.isin(dart_a_list+dart_d_list)), "blood_point"] +1
        df.loc[(df.父=="スマートファルコン ") & (df.course.isin(dart_c_list)), "blood_point"] = \
            df.loc[(df.父=="スマートファルコン ") & (df.course.isin(dart_c_list)), "blood_point"] +1
        df.loc[(df.父=="ドゥラメンテ ") & (df.course.isin(turf_a_list+turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="ドゥラメンテ ") & (df.course.isin(turf_a_list+turf_d_list)), "blood_point"] +2
        df.loc[(df.父=="ドゥラメンテ ") & (df.course.isin(turf_b_list)), "blood_point"] = \
            df.loc[(df.父=="ドゥラメンテ ") & (df.course.isin(turf_b_list)), "blood_point"] -1
        df.loc[(df.父=="ドゥラメンテ ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] = \
            df.loc[(df.父=="ドゥラメンテ ") & (df.sex=="牡") & (df.race_type=="ダート"), "blood_point"] +1
        df.loc[(df.父=="モーリス ") & (((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] = \
            df.loc[(df.父=="モーリス ") & (((df.sex=="牡") & (df.course.isin(turf_b_list+turf_c_list))) | ((df.sex=="牝") & (df.course.isin(turf_a_list)))), "blood_point"] +2
        df.loc[(df.父=="モーリス ") & (df.course.isin(turf_d_list)), "blood_point"] = \
            df.loc[(df.父=="モーリス ") & (df.course.isin(turf_d_list)), "blood_point"] -1
        df.loc[(df.父=="ダノンレジェンド ") & (df.course.isin(dart_a_list+dart_b_list)), "blood_point"] = \
            df.loc[(df.父=="ダノンレジェンド ") & (df.course.isin(dart_a_list+dart_b_list)), "blood_point"] +2
        df.loc[((df.母父_米国型=="1") | (df.母母父_米国型=="1") | (df.非サンデー馬=="1")) & (df.course.isin(turf_a_list)), "blood_point"] =\
            df.loc[((df.母父_米国型=="1") | (df.母母父_米国型=="1") | (df.非サンデー馬=="1")) & (df.course.isin(turf_a_list)), "blood_point"] +1
        df.loc[((df.父_米国型=="1") | (df.父_欧州型=="1") | (df.非サンデー馬=="1") | (df.母父_日本型=="1")) & (df.course.isin(turf_b_list)), "blood_point"] =\
            df.loc[((df.母父_米国型=="1") | (df.母母父_米国型=="1") | (df.非サンデー馬=="1") | (df.母父_日本型=="1")) & (df.course.isin(turf_b_list)), "blood_point"] +1
        df.loc[((df.欧州A=="1") | (df.欧州B=="1")) & (df.course.isin(turf_c_list)), "blood_point"] =\
            df.loc[((df.欧州A=="1") | (df.欧州B=="1")) & (df.course.isin(turf_c_list)), "blood_point"] +1
        df.loc[((df.母父_欧州型=="1") | (df.母母父_欧州型=="1")) & (df.父_日本型=="1") & (df.course.isin(turf_d_list)), "blood_point"] =\
            df.loc[((df.母父_欧州型=="1") | (df.母父_欧州型=="1")) & (df.父_日本型=="1") & (df.course.isin(turf_d_list)), "blood_point"] +1
        df.loc[((df.米国A=="1") | (df.米国B=="1") | (df.非サンデー馬=="1")) & (df.course.isin(dart_a_list)), "blood_point"] =\
            df.loc[((df.米国A=="1") | (df.米国B=="1") | (df.父_米国型=="1")) & (df.course.isin(dart_a_list)), "blood_point"] +1
        df.loc[((df.欧州A=="1") | (df.父_母父_ディープ=="1")) & (df.course.isin(dart_a_list)), "blood_point"] =\
            df.loc[((df.欧州A=="1") | (df.父_母父_ディープ=="1")) & (df.course.isin(dart_a_list)), "blood_point"] -1
        df.loc[((df.父_母父_ディープ=="1") | (df.父_母父_キングマンボ系=="1") | (df.母父_大系統ナスルーラ=="1")) & (df.course.isin(dart_b_list)), "blood_point"] =\
            df.loc[((df.父_母父_ディープ=="1") | (df.父_母父_キングマンボ系=="1") | (df.母父_大系統ナスルーラ=="1")) & (df.course.isin(dart_b_list)), "blood_point"] +1
        df.loc[((df.父_日本型=="1") | (df.父_欧州型=="1") | (df.母父_ニジンスキー系=="1")) & (df.course.isin(dart_c_list)), "blood_point"] =\
            df.loc[((df.父_日本型=="1") | (df.父_欧州型=="1") | (df.母父_ニジンスキー系=="1")) & (df.course.isin(dart_c_list)), "blood_point"] +1
        df.loc[((df.欧州B=="1") | (df.父_大系統ナスルーラ=="1") | (df.父_サドラー系=="1")) & (df.course.isin(dart_d_list)), "blood_point"] =\
            df.loc[((df.欧州B=="1") | (df.父_大系統ナスルーラ=="1") | (df.父_サドラー系=="1")) & (df.course.isin(dart_d_list)), "blood_point"] +1
        df.loc[(df.sex=="牝") & (df.course.isin(turf_c_list)), "blood_point"] =\
            df.loc[(df.sex=="牝") & (df.course.isin(turf_c_list)), "blood_point"] -1
        df.loc[(df.sex=="牝") & (df.course.isin(dart_d_list)), "blood_point"] =\
            df.loc[(df.sex=="牝") & (df.course.isin(dart_d_list)), "blood_point"] -1
        pace_down_list = ["ディープインパクト ", "エピファネイア ", "ルーラーシップ ", "パイロ Pyro(米) ", "トゥザグローリー ",\
                         "ノヴェリスト Novellist(愛) ", "ヨハネスブルグ Johannesburg(米) ", "ハービンジャー Harbinger(英) ",\
                         "ネオユニヴァース ", "ワールドエース ", "ジャスタウェイ "]
        pace_up_list = ["キングカメハメハ ", "ロードカナロア ", "ドゥラメンテ ", "サウスヴィグラス ", "カジノドライヴ ",\
                       "アイルハヴアナザー I\'ll Have Another(米) ", "エンパイアメーカー Empire Maker(米) ", "ベルシャザール ",\
                       "エスケンデレヤ ", "タートルボウル Turtle Bowl(愛) ", "スクリーンヒーロー ", "モーリス "]
        df.loc[(df.今回距離変化 < 0) & (df.父.isin(pace_down_list)), "blood_point"] = \
            df.loc[(df.今回距離変化 < 0) & (df.父.isin(pace_down_list)), "blood_point"] +2
        df.loc[(df.今回距離変化 < 0) & (df.父.isin(pace_up_list)), "blood_point"] = \
            df.loc[(df.今回距離変化 < 0) & (df.父.isin(pace_up_list)), "blood_point"] -2
        df.loc[(df.今回距離変化 > 0) & (df.父.isin(pace_up_list)), "blood_point"] = \
            df.loc[(df.今回距離変化 > 0) & (df.父.isin(pace_up_list)), "blood_point"] +2
        df.loc[(df.今回距離変化 > 0) & (df.父.isin(pace_down_list)), "blood_point"] = \
            df.loc[(df.今回距離変化 > 0) & (df.父.isin(pace_down_list)), "blood_point"] -2
        df.loc[(df.前走距離変化 > 0) & (df.今回距離変化 == 0) & (df.父.isin(pace_down_list)), "blood_point"] = \
            df.loc[(df.前走距離変化 > 0) & (df.今回距離変化 == 0) & (df.父.isin(pace_down_list)), "blood_point"] +1
        df.loc[(df.今回間隔 > "65 days") & (df.父.isin(pace_down_list)), "blood_point"] =\
            df.loc[(df.今回間隔 > "65 days") & (df.父.isin(pace_down_list)), "blood_point"] +1
        df.loc[(df.前走間隔 > "65 days") & (df.着順_1走前 <= 5) & (df.今回間隔 < "40 days") & (df.父.isin(pace_down_list)), "blood_point"] =\
            df.loc[(df.前走間隔 > "65 days") & (df.着順_1走前 <= 5) & (df.今回間隔 < "40 days") & (df.父.isin(pace_down_list)), "blood_point"] -1
        df.loc[(df.今回間隔 > "65 days") & (df.父.isin(pace_up_list)), "blood_point"] =\
            df.loc[(df.今回間隔 > "65 days") & (df.父.isin(pace_up_list)), "blood_point"] -1
        df.loc[(df.前走間隔 > "65 days") & (df.着順_1走前 <= 9) & (df.今回間隔 < "40 days") & (df.父.isin(pace_up_list)), "blood_point"] =\
            df.loc[(df.前走間隔 > "65 days") & (df.着順_1走前 <= 9) & (df.今回間隔 < "40 days") & (df.父.isin(pace_up_list)), "blood_point"] +1
        
        df["advantage_point"] = 0
        df.loc[(df.前走距離変化 > 0) & (df.着順_1走前 <=3), "advantage_point"] =\
            df.loc[(df.前走距離変化 > 0) & (df.着順_1走前 <=3), "advantage_point"] -2
        df.loc[(df.前々走距離変化 > 0) & (df.着順_2走前 <= 3) & (df.着順_1走前 >= 5), "advantage_point"] =\
            df.loc[(df.前々走距離変化 > 0) & (df.着順_2走前 <= 3) & (df.着順_1走前 >= 5), "advantage_point"] +1
        dart_gate_a_plus = ["01ダート1700", "02ダート1700", "05ダート1300", "05ダート1600", "07ダート1200", "07ダート1800"]
        dart_gate_b_plus = ["02ダート1700", "05ダート1300", "07ダート1800"]
        dart_gate_c_plus = ["05ダート1400", "05ダート2100", "07ダート1200"]
        dart_gate_d_plus = ["05ダート2100", "09ダート1200", "09ダート1400"]
        dart_gate_a_minus = ["03ダート1150", "03ダート1700", "04ダート1200", "05ダート2100", "07ダート1400", "08ダート1200",\
                             "08ダート1400", "08ダート1800", "09ダート1200", "09ダート1400", "10ダート1000", "10ダート1700"]
        dart_gate_c_minus = ["04ダート1800", "07ダート1800", "10ダート1000"]
        dart_gate_d_minus = ["05ダート1600", "06ダート1800", "07ダート1200", "07ダート1800", "10ダート1700"]
        df.loc[(df.馬番 <= 4) & (df.course.isin(dart_gate_a_plus)), "advantage_point"] =\
            df.loc[(df.馬番 <= 4) & (df.course.isin(dart_gate_a_plus)), "advantage_point"] +1
        df.loc[(df.馬番 <= 9) & (df.馬番 >= 5) & (df.course.isin(dart_gate_b_plus)), "advantage_point"] =\
            df.loc[(df.馬番 <= 9) & (df.馬番 >= 5) & (df.course.isin(dart_gate_b_plus)), "advantage_point"] +1
        df.loc[(df.馬番 <= 14) & (df.馬番 >= 10) & (df.course.isin(dart_gate_c_plus)), "advantage_point"] =\
            df.loc[(df.馬番 <= 14) & (df.馬番 >= 10) & (df.course.isin(dart_gate_c_plus)), "advantage_point"] +1
        df.loc[(df.馬番 >= 15) & (df.course.isin(dart_gate_d_plus)), "advantage_point"] =\
            df.loc[(df.馬番 >= 15) & (df.course.isin(dart_gate_d_plus)), "advantage_point"] +1
        df.loc[(df.馬番 <= 4) & (df.course.isin(dart_gate_a_minus)), "advantage_point"] =\
            df.loc[(df.馬番 <= 4) & (df.course.isin(dart_gate_a_minus)), "advantage_point"] -1
        df.loc[(df.馬番 <= 14) & (df.馬番 >= 10) & (df.course.isin(dart_gate_c_minus)), "advantage_point"] =\
            df.loc[(df.馬番 <= 14) & (df.馬番 >= 10) & (df.course.isin(dart_gate_c_minus)), "advantage_point"] -1
        df.loc[(df.馬番 >= 15) & (df.course.isin(dart_gate_d_minus)), "advantage_point"] =\
            df.loc[(df.馬番 >= 15) & (df.course.isin(dart_gate_d_minus)), "advantage_point"] -1
        df.loc[(df.race_type == "芝") & (df.馬番_1走前 <=5) & (df.着順_1走前 <=4) & (df.馬番 >= 14), "advantage_point"] =\
            df.loc[(df.race_type == "芝") & (df.馬番_1走前 <=5) & (df.着順_1走前 <=4) & (df.馬番 >= 14), "advantage_point"] -1
        df.loc[(df.race_type == "ダート") & (df.馬番 <=5) & (df.着順_1走前 <=3 ) & (df.馬番_1走前 >=14), "advantage_point"] =\
            df.loc[(df.race_type == "ダート") & (df.馬番 <=5) & (df.着順_1走前 <=3 ) & (df.馬番_1走前 >=14), "advantage_point"] -1
        df.loc[(df.first_corner_1走前==1) & (df.着順_1走前==1), "advantage_point"] =\
            df.loc[(df.first_corner_1走前==1) & (df.着順_1走前==1), "advantage_point"] -1
        df["実績季節"] = df["実績季節"].fillna("無し")
        df.loc[df.apply(lambda x: x.季節 in x.実績季節, axis=1), "advantage_point"] =\
            df.loc[df.apply(lambda x: x.季節 in x.実績季節, axis=1), "advantage_point"] +1
        
        self.merged_df = df.copy()
        
    def point(self):
        df = self.merged_df.copy()
        df["jockey_point"] = 0
        df["trainer_point"] = 0
        plus_trainer_list = ["01148","01092","01075","00438","01105","01086","01002","01126"]
        plus_jockey_list = ["01126","05339","01014","01075","01088","01102","01093","05386","00666"]
        df.loc[df.jockey_id.isin(plus_jockey_list), "jockey_point"] =\
            df.loc[df.jockey_id.isin(plus_jockey_list), "jockey_point"] +2
        df.loc[df.trainer_id.isin(plus_trainer_list), "trainer_point"] =\
            df.loc[df.trainer_id.isin(plus_trainer_list), "trainer_point"] +2
        df["実績騎手"] = df["実績騎手"].fillna("無し")
        df.loc[df.apply(lambda x: x.jockey_id in x.実績騎手, axis=1), "jockey_point"] =\
            df.loc[df.apply(lambda x: x.jockey_id in x.実績騎手, axis=1), "jockey_point"] +2

        df["着差_point"] = 0
        df.loc[df.着差_1走前 <= (-0.3), "着差_point"] =\
            df.loc[df.着差_1走前 <= (-0.3), "着差_point"] +3
        df.loc[(df.着差_1走前 <= 0) & (df.着差_1走前 >= (-0.2)), "着差_point"] =\
            df.loc[(df.着差_1走前 <= 0) & (df.着差_1走前 >= (-0.2)), "着差_point"] +2
        df.loc[(df.着差_1走前 <= 0.2) & (df.着差_1走前 >= (0.1)), "着差_point"] =\
            df.loc[(df.着差_1走前 <= 0.2) & (df.着差_1走前 >= (0.1)), "着差_point"] +1
        df.loc[(df.着差_1走前 <= 0.6) & (df.着差_1走前 >= (0.3)), "着差_point"] =\
            df.loc[(df.着差_1走前 <= 0.6) & (df.着差_1走前 >= (0.3)), "着差_point"] +0
        df.loc[(df.着差_1走前 <= 0.9) & (df.着差_1走前 >= (0.7)), "着差_point"] =\
            df.loc[(df.着差_1走前 <= 0.9) & (df.着差_1走前 >= (0.7)), "着差_point"] -1
        df.loc[df.着差_1走前 >= 1, "着差_point"] =\
            df.loc[df.着差_1走前 >= 1, "着差_point"] -2

        df["上り_1走前"].fillna(0, inplace=True)
        df["上り_2走前"].fillna(0, inplace=True)
        df["上り_3走前"].fillna(0, inplace=True)
        df["上り_ave"] = (df["上り_1走前"] + df["上り_2走前"] + df["上り_3走前"]) / 3

        df["着順_point"] = 0
        df.loc[(df.着順_1走前 >= 4) & (df.着順_1走前 <= 8), "着順_point"] =\
            df.loc[(df.着順_1走前 >= 4) & (df.着順_1走前 <= 8), "着順_point"] +1
        df.loc[(df.実質着順_2走前 >= 0.8), "着順_point"] =\
            df.loc[(df.実質着順_2走前 >= 0.8), "着順_point"] +3
        df.loc[(df.実質着順_2走前 >= 0.7) & (df.実質着順_2走前 < 0.8), "着順_point"] =\
            df.loc[(df.実質着順_2走前 >= 0.7) & (df.実質着順_2走前 < 0.8), "着順_point"] +2
        df.loc[(df.実質着順_2走前 >= 0.6) & (df.実質着順_2走前 < 0.7), "着順_point"] =\
            df.loc[(df.実質着順_2走前 >= 0.6) & (df.実質着順_2走前 < 0.7), "着順_point"] +1
        df.loc[(df.実質着順_2走前 >= 0.3) & (df.実質着順_2走前 < 0.4), "着順_point"] =\
            df.loc[(df.実質着順_2走前 >= 0.3) & (df.実質着順_2走前 < 0.4), "着順_point"] -1
        df.loc[(df.実質着順_2走前 < 0.3), "着順_point"] =\
            df.loc[(df.実質着順_2走前 < 0.3), "着順_point"] -2
        df.loc[(df.実質着順_3走前 >= 0.8), "着順_point"] =\
            df.loc[(df.実質着順_3走前 >= 0.8), "着順_point"] +3
        df.loc[(df.実質着順_3走前 >= 0.7) & (df.実質着順_3走前 < 0.8), "着順_point"] =\
            df.loc[(df.実質着順_3走前 >= 0.7) & (df.実質着順_3走前 < 0.8), "着順_point"] +2
        df.loc[(df.実質着順_3走前 >= 0.6) & (df.実質着順_3走前 < 0.7), "着順_point"] =\
            df.loc[(df.実質着順_3走前 >= 0.6) & (df.実質着順_3走前 < 0.7), "着順_point"] +1
        df.loc[(df.実質着順_3走前 >= 0.3) & (df.実質着順_3走前 < 0.4), "着順_point"] =\
            df.loc[(df.実質着順_3走前 >= 0.3) & (df.実質着順_3走前 < 0.4), "着順_point"] -1
        df.loc[(df.実質着順_3走前 < 0.3), "着順_point"] =\
            df.loc[(df.実質着順_3走前 < 0.3), "着順_point"] -2
        
        df["blood_point"].fillna(0, inplace=True)
        df["camp"] = df["jockey_point"] + df["trainer_point"]
        df = pd.merge(df, df.groupby("race_id")[["blood_point", "着差_point", "着順_point", "camp", "上り_ave"]].mean().rename(columns={"blood_point":"blood_point_race_ave",\
                                        "着差_point":"着差_race_ave", "着順_point":"着順_race_ave", "camp":"camp_race_ave", "上り_ave":"上り_race_ave"}), on="race_id", how="left")
        df = pd.merge(df, df.groupby("race_id")[["blood_point", "着差_point", "着順_point", "camp", "上り_ave"]].std().rename(columns={"blood_point":"blood_point_race_std",\
                                        "着差_point":"着差_race_std", "着順_point":"着順_race_std", "camp":"camp_race_std", "上り_ave":"上り_race_std"}), on="race_id", how="left")
        df["blood_point_race_std"] = df["blood_point_race_std"].replace(0)
        df["bld_point"] = df.apply(lambda x: (x.blood_point - x.blood_point_race_ave) / x.blood_point_race_std, axis=1)
        df["着差_race_std"] = df["着差_race_std"].replace(0)
        df["lead_point"] = df.apply(lambda x: (x.着差_point - x.着差_race_ave) / x.着差_race_std, axis=1)
        df["着順_race_std"] = df["着順_race_std"].replace(0)
        df["rank_point"] = df.apply(lambda x: (x.着順_point - x.着順_race_ave) / x.着順_race_std, axis=1)
        df["camp_race_std"] = df["camp_race_std"].replace(0)
        df["camp_point"] = df.apply(lambda x: (x.camp - x.camp_race_ave) / x.camp_race_std, axis=1)
        df["上り_race_std"] = df["上り_race_std"].replace(0)
        df["agari_point"] = df.apply(lambda x: (x.上り_ave - x.上り_race_ave) / x.上り_race_std, axis=1)
        df["point_all"] = df["bld_point"] + df["lead_point"] + df["rank_point"] + df["agari_point"] + df["camp_point"] + df["advantage_point"]
        
        df = df.set_index("race_id")
        df = df[['着順', 'race_type', '馬番', 'bld_point', 'lead_point', 'rank_point', 'agari_point', 'camp_point', 'advantage_point', 'point_all']]
        
        self.pointed = df.copy()
    

df = Peds.pointed.copy()
df["quinella"] = df["着順"].map(lambda x: 1 if x <= 2 else 0)
train_data = df.drop(columns={"quinella", "馬番"})
y = df_quinella["quinella"]
X = train_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

clf = RandomForestClassifier(random_state=1234)
clf.fit(X_train, y_train)
fi = clf.feature_importances_
print("score=", clf.score(X_test, y_test))
idx = np.argsort(fi)[::-1]
top_cols, top_importances = X_train.columns.values[idx], fi[idx]
print("random forest importance")
print(top_cols, top_importances)