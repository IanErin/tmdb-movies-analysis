# 导入所需数据库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast # 安全解析为python对象


# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def data_preparation():

    # 导入数据
    movies_df = pd.read_csv("..\\data\\tmdb_5000_movies.csv")
    credits_df = pd.read_csv("..\\data\\tmdb_5000_credits.csv")

    # 合并数据
    credits_df = credits_df[["movie_id","cast","crew"]]
    movies_df = movies_df.merge(credits_df,left_on="id",right_on="movie_id")

    # 根据分析目标选择需要的列
    key_columns = ["id","title","genres","keywords","release_date","runtime","budget","revenue","original_language","vote_average",
                   "vote_count","cast","crew","overview","production_companies","production_countries","popularity"]
    movies_df = movies_df[key_columns]

    # 填充runtime的缺失值
    movies_df["runtime"] = movies_df["runtime"].fillna(movies_df["runtime"].median())
    # 删除release_date缺失的行
    movies_df = movies_df.dropna(subset=["release_date"])

    # 处理异常值 对不合理的数据修改，比如预算和票房为0的视为缺失值
    movies_df = movies_df[(movies_df["budget"]>1000) & (movies_df["revenue"]>1000)]
    # movies_df = movies_df[movies_df["vote_count"] >= 0]

    # 修改日期数据类型，创建年月相关的列，方便按年分析
    movies_df["release_date"] = pd.to_datetime(movies_df["release_date"]) # 讲日期转化为datetime类型
    movies_df["release_year"] = movies_df["release_date"].dt.year # 提取上映年份，方便按年分析
    movies_df["release_month"] = movies_df["release_date"].dt.month # 提取月份

    # 计算盈利能力（profit），和投资回报率（ROI）
    movies_df["profit"] = movies_df["revenue"] - movies_df["budget"]
    movies_df["roi"] = (movies_df["profit"]/movies_df["budget"])*100 # 求百分比

    # 处理RIO异常值
    movies_df.loc[movies_df["budget"] <= 0,'roi'] = np.nan
    movies_df["roi"] = movies_df["roi"].replace([np.inf,-np.inf] , np.nan)

    # 解析JSON格式列 (genres, keywords, cast, crew, production_companies, production_countries)
    # 这些列存储的是字符串形式的JSON列表，需要解析成Python对象（列表/字典）
    def parse_json_column(column):
        try:
            return ast.literal_eval(column)
        except (ValueError,SyntaxError):
            return [] # 失败返回空列表

    json_columns = ["genres","keywords","cast","crew","production_companies","production_countries"]
    for col in json_columns:
        movies_df[col] = movies_df[col].apply(parse_json_column)

    # 从crew列提取导演信息
    def get_director(crew_list):
        for person in crew_list:
            if person["job"] == "Director":
                return person["name"]
        return np.nan

    movies_df["director"] = movies_df["crew"].apply(get_director)

    # 从 genres 列提取电影类型
    def get_genres_list(genre_list):
        if isinstance(genre_list,list):
            return [genre["name"] for genre in genre_list]
        else:
            return []

    movies_df["genres_list"] = movies_df["genres"].apply(get_genres_list)

    # 选取1960年及以后的电影数据
    movies_df = movies_df[movies_df['release_year'] >= 1960]
    return movies_df

# 准备数据
df = data_preparation()

# 假设电影预算<2000万,票房>1亿的成为黑马电影
heima_count = df[(df["budget"]<20000000) & (df["revenue"]>100000000)]["id"].count()
movies_count = df["id"].count()
print("黑马电影数为：",heima_count)
print("黑马电影比例为：",(heima_count/movies_count)*100,"%")

# 计算预算和票房的年化增速
annual_budget = df.groupby("release_year")["budget"].sum()
first_year_budget = annual_budget.iloc[0]
last_year_budget = annual_budget.iloc[-1]
years = len(annual_budget.index)
budget_cagr = (last_year_budget/first_year_budget) ** (1/years) - 1
print("预算年化增速为：",budget_cagr)

annual_revenue = df.groupby("release_year")["revenue"].sum()
first_year_revenue = annual_revenue.iloc[0]
last_year_revenue = annual_revenue.iloc[-1]
years = len(annual_revenue.index)
revenue_cagr = (last_year_revenue/first_year_revenue) ** (1/years) - 1
print("票房年化增速为：",revenue_cagr)

def genres_selection(data_df):
    selection_df = data_df
    all_genres = []
    for genres in selection_df["genres_list"]:
        all_genres.extend(genres)
    return pd.Series(all_genres).value_counts()

median_budget_movie_df = df[(df["budget"]>=30000000) & (df["budget"]<100000000)]
median_budget_genres = genres_selection(median_budget_movie_df)
median_budget_high_roi_movie_df = median_budget_movie_df[median_budget_movie_df["roi"]>20]
median_budget_high_roi_genres =  genres_selection(median_budget_high_roi_movie_df)
for i,j in median_budget_high_roi_genres.items():
    k = median_budget_genres[i]
    median_budget_genre_roi = j / k * 100
    print("{} 收益率大于20%的电影占比为：{}".format(i,median_budget_genre_roi))

# 对于预算大于1亿的大制作，看哪些导演ROI更高
high_budget_roi = df[df["budget"]>100000000].groupby("director")["roi"].median().sort_values(ascending=False).dropna().head()
print("大制作电影高收益导演名单 (前5)：")
print(high_budget_roi)

# 假设执导过票房5亿及以上的导演为顶级导演，判断顶级导演对大制作电影的提升
high_budget_movies = df[df["budget"]>100000000].copy()

# 假设执导过票房高于5亿的导演为顶级导演
top_director = df.groupby("director")["revenue"].max() > 500000000
top_director = top_director[top_director].index.tolist()

# 添加新的一列区分顶级导演和非顶级导演
high_budget_movies["group"] = high_budget_movies['director'].apply(lambda x: 'Top Director' if x in top_director else 'Other Director')

# 计算各组的平均ROI
grouped_roi = high_budget_movies.groupby("group")["roi"].mean()

# 输出提升百分比
top_director_roi = grouped_roi['Top Director']
other_director_roi = grouped_roi['Other Director']
improvement_percentage = ((top_director_roi - other_director_roi) / other_director_roi) * 100
print("顶级导演对大制作电影的ROI可以提升：",improvement_percentage)