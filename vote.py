from util import *

# df1 = load_data_from_csv("result/textCNN2_result 0.6308.csv")
# df2 = load_data_from_csv("result/binaryTextCNN_result 0.61.csv")
# df3 = load_data_from_csv("result/fastText_result 0.54.csv")
#
# columns = df1.columns.values.tolist()
# result_df = pd.DataFrame({columns[0]: [x for x in range(len(df1))]})
# result_df['content'] = ''  # 省略content
#
# for i in range(20):
# 	vote_predictions = []
# 	column = columns[i+2]
# 	for i in range(len(df1)):
# 		if df1[column][i] != -2:
# 			vote_predictions.append(df1[column][i])
# 		elif df2[column][i] != -2:
# 			vote_predictions.append(df2[column][i])
# 		elif df3[column][i] != -2:
# 			vote_predictions.append(df3[column][i])
# 		else:
# 			vote_predictions.append(-2)
# 	result_df[column]=vote_predictions
#
# result_df.to_csv('result/vote_result.csv', index=False, encoding="utf-8")


df1 = load_data_from_csv("result/result_b.csv")
df2 = load_data_from_csv("result/result.csv")

columns = df1.columns.values.tolist()
column = columns[7]
df1[column] = df2[column]
df1.to_csv("result/result_B.csv",index=False, encoding="utf-8")