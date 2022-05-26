import pandas as pd

fname = "b1"
df = pd.read_csv(fname + "_f.csv")

rm1 = df.iloc[214:235, :]
# rm2 = df.iloc[1395:1564, :]
frames = [rm1]
rm = pd.concat(frames)
lst = ["rm" for x in range(0,rm.shape[0])]
rm["res"] = lst

lm = df.iloc[535:600, :]
lst = ["lm" for x in range(0,lm.shape[0])]
lm["res"] = lst

cm = df.iloc[867:1000, :]
lst = ["cm" for x in range(0,cm.shape[0])]
cm["res"] = lst

# med = df.iloc[1738:2064, :]
# lst = ["med" for x in range(0,med.shape[0])]
# med["res"] = lst

sf = df.iloc[5:190, :]
# sf1 = df.iloc[737:850, :]
# sf2 = df.iloc[1530:1675, :]
# sf3 = df.iloc[1930:2030, :]
frames = [sf]
front = pd.concat(frames)
lst = ["sf" for x in range(0,front.shape[0])]
front["res"] = lst

ra = df.iloc[1415:1450, :]
# ra1 = df.iloc[1342:1417, :]
# ra2 = df.iloc[1460:1540, :]
# ra3 = df.iloc[1610:1750, :]
rframes = [ra]
rand = pd.concat(rframes)
lst = ["eor" for x in range(0,rand.shape[0])]
rand["res"] = lst


rm.to_csv("rm/" + fname +'_rm.csv', header=True, index=False)
lm.to_csv("lm/" + fname +'_lm.csv', header=True, index=False)
# cm.to_csv("cm/" + fname +'_cm.csv', header=True, index=False)
front.to_csv("sf/" + fname +'_sf.csv', header=True, index=False)
# med.to_csv("med/" + fname +'_med.csv', header=True, index=False)
rand.to_csv("rand/" + fname +'_rand.csv', header=True, index=False)
