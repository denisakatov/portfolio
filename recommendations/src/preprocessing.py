import numpy as np
import pandas as pd


def train_test_preprocessing(data, train, recommender, N):
    # Подготовка обучающего датасета, разбиение на X и y

    users_lvl_2 = pd.DataFrame(data['user_id'].unique())

    users_lvl_2.columns = ['user_id']

    train_users = train['user_id'].unique()
    users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]

    # Рекомендации на основе собственных покупок
    users_lvl_2_ = users_lvl_2.copy()
    users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(lambda x: recommender.get_own_recommendations(x, N=N))

    s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)

    s.name = 'item_id'

    users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)

    users_lvl_2['flag'] = 1

    targets_lvl_2 = data[['user_id', 'item_id']].copy()
    targets_lvl_2.head(2)

    targets_lvl_2['target'] = 1  # только покупки

    targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')

    targets_lvl_2['target'].fillna(0, inplace=True)
    
    targets_lvl_2.drop('flag', axis=1, inplace=True)

    return targets_lvl_2



def new_item_features(data, item_features, items_emb_df):
    # признаки для продуктов
    new_item_features = item_features.merge(data, on='item_id', how='left')

    # Скидки
    mean_disc = new_item_features.groupby('item_id')['coupon_disc'].mean().reset_index().sort_values('coupon_disc')
    item_features = item_features.merge(mean_disc, on='item_id', how='left')
    
    # Добавим эмбеддинги
    item_features = item_features.merge(items_emb_df, how='left')


    return item_features


def new_user_features(data, user_features, users_emb_df):
    # Новые признаки для пользователей
    data['price']=data['sales_value']/data['quantity']
    new_user_features = user_features.merge(data, on='user_id', how='left')


    # Добавим имбеддинги
    user_features = user_features.merge(users_emb_df, how='left')


    # Обычное время покупки
    time = new_user_features.groupby('user_id')['trans_time'].mean().reset_index()
    time.rename(columns={'trans_time': 'mean_time'}, inplace=True)
    time = time.astype(np.float32)
    user_features = user_features.merge(time, how='left')


    # Возраст
    user_features['age'] = user_features['age_desc'].replace({'65+': 65, '45-54': 45, '25-34': 25, '35-44': 35, '19-24':19, '55-64':55}
    )
    user_features = user_features.drop('age_desc', axis=1)


    # Доход
    user_features['income'] = user_features['income_desc'].replace(
        {'Under 15K': 15,'15-24K': 24,'35-49K': 35,'25-34K': 25,'50-74K': 50,'75-99K': 75,'100-124K': 120, '125-149K': 125, '150-174K': 150,  '250K+': 250, '175-199K': 175,'200-249K': 200})
    user_features = user_features.drop('income_desc', axis=1)


    # Дети
    user_features['kids'] = 0
    user_features.loc[(user_features['kid_category_desc'] == '1'), 'kids'] = 1
    user_features.loc[(user_features['kid_category_desc'] == '2'), 'kids'] = 2
    user_features.loc[(user_features['kid_category_desc'] == '3'), 'kids'] = 3
    user_features = user_features.drop('kid_category_desc', axis=1)


    # Средний чек, средний чек в неделю
    basket = new_user_features.groupby(['user_id'])['price'].sum().reset_index()

    baskets_qnt = new_user_features.groupby('user_id')['basket_id'].count().reset_index()
    baskets_qnt.rename(columns={'basket_id': 'baskets_qnt'}, inplace=True)

    average_basket = basket.merge(baskets_qnt)

    average_basket['average_basket'] = average_basket.price / average_basket.baskets_qnt
    average_basket['sum_per_week'] = average_basket.price / new_user_features.week_no.nunique()

    average_basket = average_basket.drop(['price', 'baskets_qnt'], axis=1)
    user_features = user_features.merge(average_basket, how='left')

    return user_features








def new_features(data, train, recommender, item_features, user_features, items_emb_df, users_emb_df, N=50):

    target = train_test_preprocessing(data, train, recommender, N)
    user_features = new_user_features(data, user_features, users_emb_df)
    item_features = new_item_features(data, item_features, items_emb_df)
    item_features = data.merge(item_features, on='item_id', how='left')

     

    data_df = item_features.merge(user_features, on='user_id', how='left')

    
    # отношение количества покупок товаров в данной категории к среднему количеству
    df_1 = data_df.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'quantity': 'mean'}).reset_index().rename(columns={'quantity': 'count_purchases_week_dep'})

    df_2 = data_df.groupby(['commodity_desc', 'week_no']).agg({'quantity': 'sum'}).reset_index().rename(columns=({'quantity': 'mean_count_purchases_week_dep'}))

    df = df_1.merge(df_2, on=['commodity_desc', 'week_no'], how='left') # объединяем группе товаров и неделям.
    df['count_purchases_week_mean'] = df['count_purchases_week_dep'] / df['mean_count_purchases_week_dep']
    df = df[['user_id', 'commodity_desc', 'count_purchases_week_mean']]

    temp_df = df.groupby(['user_id', 'commodity_desc']).agg({'count_purchases_week_mean': 'mean'}).reset_index()

    data_df = data_df.merge(temp_df, on=['user_id', 'commodity_desc'], how='left')

    
    
    # отношение суммы покупок товаров к средней сумме
    df_1 = data_df.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'price': 'sum'}).reset_index().rename(columns={'price': 'price_week'})

    df_2 = data_df.groupby(['commodity_desc', 'week_no']).agg({'price': 'sum'}).reset_index().rename(columns=({'price': 'mean_price_week'}))

    df = df_1.merge(df_2, on=['commodity_desc', 'week_no'], how='left')
    df['sum_purchases_week_mean'] = df['price_week'] / df['mean_price_week']
    df = df[['user_id', 'commodity_desc', 'sum_purchases_week_mean']]

    temp_df = df.groupby(['user_id', 'commodity_desc']).agg({'sum_purchases_week_mean': 'mean'}).reset_index()

    data_df = data_df.merge(temp_df, on=['user_id', 'commodity_desc'], how='left')

    data_df = data_df.merge(target, on=['item_id', 'user_id'], how='left')
    data_df = data_df.fillna(0)

    return data_df




    # Список важных признаков
def get_important_features(model, X_train, y_train):

    model.fit(X_train, y_train)
    
    feature_imp = list(zip(X_train.columns.tolist(), model.feature_importances_))
    feature_imp = pd.DataFrame(feature_imp, columns=['feature', 'value'])
    basic_feats = feature_imp.loc[feature_imp.value > 0, 'feature'].tolist()
    return basic_feats

 #  Получение списка товаров из уникальных категорий

def get_different_categories(list_recommendations, item_info):
    finish_recommendations = []

    categories_used = []

    CATEGORY_NAME = 'sub_commodity_desc'

    for item in list_recommendations:
        category = item_info.loc[item_info['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            finish_recommendations.append(item)
            categories_used.append(category)

    return finish_recommendations


# Топ-n популярных товаров

def popularity_recommendation(data, n=5):

    popular = data.groupby('item_id')['quantity'].count().reset_index()
    popular.sort_values('quantity', ascending=False, inplace=True)
    popular = popular[popular['item_id'] != 999999]
    recs = popular.head(n).item_id
    return recs.tolist()




def get_final_recomendation_list(row, item_info, train_1, df_price, list_pop_rec, N=5):

    recommend = row['recomendations']
    purchased_goods = train_1.loc[train_1['user_id'] == row['user_id']]['item_id'].unique()

    if recommend == 0:
        recommend = list_pop_rec

    # 1 Уникальность
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommend if item not in unique_recommendations]

    # 2 Товары дороже 1$
    price_recommendations = []
    [price_recommendations.append(item) for item in unique_recommendations if df_price.loc[df_price['item_id'] == item]['price'].values > 1]

    # 3 один товар > 7 $
    expensive_items = []
    [expensive_items.append(item) for item in price_recommendations if df_price.loc[df_price['item_id'] == item]['price'].values > 7]

    if len(expensive_items) ==0:
        [expensive_items.append(item) for item in list_pop_rec if df_price.loc[df_price['item_id'] == item]['price'].values > 7]

    # 4 товара юзер никогда не покупал
    new_items = []
    [new_items.append(item) for item in price_recommendations if item not in purchased_goods]

    # Промежуточный итог
    support_rec = []
    support_rec.append(expensive_items[0] if len(expensive_items) > 0 else list_pop_rec[0])
    support_rec += new_items
    support_rec = get_different_categories(support_rec, item_info=item_info)[0:3]
    support_rec += price_recommendations
    final_recommendations = get_different_categories(support_rec, item_info=item_info)

    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(list_pop_rec[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]

    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations



    # Финальный список рекомендованных товаров


def get_final_recomendation(X_test, test_preds_proba, val_2, train_1, item_features):

    X_test['predict_proba'] = test_preds_proba

    X_test.sort_values(['user_id', 'predict_proba'], ascending=False, inplace=True)
    recs = X_test.groupby('user_id')['item_id']
    recomendations = []
    for user, preds in recs:
        recomendations.append({'user_id': user, 'recomendations': preds.tolist()})

    recomendations = pd.DataFrame(recomendations)

    result_2 = val_2.groupby('user_id')['item_id'].unique().reset_index()
    result_2.columns = ['user_id', 'actual']

    result = result_2.merge(recomendations, how='left')
    result['recomendations'] = result['recomendations'].fillna(0)

    df_price = train_1.groupby('item_id')['price'].mean().reset_index()

    pop_rec = popularity_recommendation(train_1, n=500)
    list_pop_rec = []
    [list_pop_rec.append(item) for item in pop_rec if df_price.loc[df_price['item_id'] == item]['price'].values > 1]

    result['recomendations'] = result.progress_apply(lambda x: get_final_recomendation_list(x, item_info=item_features, train_1=train_1, df_price=df_price, list_pop_rec=list_pop_rec, N=5), axis=1)

    return result