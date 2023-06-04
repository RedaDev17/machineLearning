import numpy as np
from graphviz import Graph

escapeFirstLine = 0
data = {}

city_categories = [["paris", "lille"], ["marseille", "toulouse"], ["madrid"]]
music_categories = [["trap", "hiphop", "rap"], ["metal", "technical death metal"], ["rock"], ["jazz"], ["classical"]]


with open('dataset.csv') as file:
    for row in file:
        if(escapeFirstLine >= 1):
            x = row.split(',')
            data[x[0]] = x
        escapeFirstLine += 1

def findIndex(stringArr, keyString):
 
    #  Initialising result array to -1
    #  in case keyString is not found
    result = []
 
    #  Iteration over all the elements
    #  of the 2-D array stored in data
 
    #  Rows
    for i in range(len(stringArr)):
 
        #  Columns
        for j in range(len(stringArr[i])):
            #  If keyString is found
            if stringArr[i][j] == keyString:
                result.append(i)
                result.append(j)
                return result
        result.append(-1)
    result.append(-1)
    #  If keyString is not found
    #  then [-1, -1] is returned
    return result

def compute_dissimilarity(user_1_id, user_2_id):
    """
    Compute  dissimilarity betwwen two users
    based on their id.

    The music_score/job_score and city_score are not a quantitative attribute.
    It is called a categorical variable.
    We must handle it differently than quantitative
    attributes.
    """
    dissimilarity = 0
    
    user_1_city_category = findIndex(city_categories, data[user_1_id][4])
    user_2_city_category = findIndex(city_categories, data[user_2_id][4])

    user_1_music_category = findIndex(music_categories, data[user_1_id][5])
    user_2_music_category = findIndex(music_categories, data[user_2_id][5])

    age_score = abs(float(data[user_1_id][1]) - float(data[user_2_id][1]))

    height_score = abs(float(data[user_1_id][2]) - float(data[user_2_id][2]))

    if (data[user_1_id][3] == data[user_2_id][3]):
        job_score = 0
    else:
        job_score = 10

    if (data[user_1_id][4] == data[user_2_id][4]):
        city_score = 0
    elif (user_1_city_category[0] == user_2_city_category[0]) :
        city_score = 5
    else:
        city_score = 10

    if (data[user_1_id][5] == "other" or data[user_2_id][5] == "other"):
        music_score = 5
    elif (data[user_1_id][5] == data[user_2_id][5]):
        music_score = 0
    else:
        music_score = abs(user_1_music_category[0] - user_2_music_category[0]) + 5

    dissimilarity = age_score ** 2 + job_score + height_score ** 2 + city_score + music_score

    print("----")
    print(
        f"user 1 {user_1_id}, user 2 {user_2_id}, dissimilarity score: {dissimilarity}"
    )
    return dissimilarity

# build a dissimilarity matrix
dissimilarity_matrix = np.zeros((199, 199))
print("compute dissimilarities")
for player_1_id in range(199):
    for player_2_id in range(199):
        dissimilarity = compute_dissimilarity(str(player_1_id), str(player_2_id))
        dissimilarity_matrix[player_1_id, player_2_id] = dissimilarity

print("dissimilarity matrix")
print(dissimilarity_matrix)

threshold = 15
# build a graph from the dissimilarity
dot = Graph(comment="Graph created from complex data", strict=True)
for player_id in range(199):
    player_name = player_id
    dot.node(str(player_name))

for player_1_id in range(199):
    # we use an undirected graph so we do not need
    # to take the potential reciprocal edge
    # into account
    for player_2_id in range(199):
        # no self loops
        if not player_1_id == player_2_id:
            player_1_name = player_1_id
            player_2_name = player_2_id
            # use the threshold condition
            # EDIT THIS LINE
            if dissimilarity_matrix[player_1_id, player_2_id] > threshold:
                dot.edge(
                    str(player_1_id),
                    str(player_2_id),
                    color="darkolivegreen4",
                    penwidth="1.1",
                )

# visualize the graph
dot.attr(label=f"threshold {threshold}", fontsize="20")
graph_name = f"images/complex_data_threshold_{threshold}"
dot.render(graph_name)