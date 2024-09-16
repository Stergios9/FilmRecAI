import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

# ------------ Task 1 ---------------
def find_unique_users_and_items(df):
    # Find the unique users and movies
    users = df['username'].unique()
    movies = df['movie'].unique()
    print("Unique Users (U):", len(users))
    print("Unique Movies (I):", len(movies))
    print("**********************************\n")

def question2(df, Rmin, Rmax, filtered_users):

    # Filter dataframe based on filtered users
    filtered_df = df[df['username'].isin(filtered_users.index)]
    print("\nfiltered_df: \n",filtered_df)
    print("\nUnique Users who have from ",Rmin," to ",Rmax," reviews are: ", len(filtered_users))

    # Find unique movies for the filtered users
    unique_movies = filtered_df['movie'].unique()
    print("\nUnique movies for users with ", Rmin, " to ", Rmax, " reviews: ", len(unique_movies),"\n")

    return filtered_df


def crowd_frequency_histogram(count_1_review,count_2_reviews,count_3_reviews):
    x_values = [202, 203,204]
    y_values = [count_1_review, count_2_reviews,count_3_reviews]

    # Plot the histogram
    plt.bar(x_values, y_values, color='skyblue', edgecolor='black')
    plt.title('Frequency Histogram of Number of Reviews per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.xticks(x_values)
    plt.grid(True)
    plt.show()

def plot_time_range_histogram(time_differences):
    # Extract time differences
    differences = [item[1] for item in time_differences]

    # Define the bins
    bins = [0, 365, 730, 1095, 1460, 1825, float('inf')]  # Bins for 1 year, 2 years, 3 years, 4 years, 5 years, and beyond

    # Calculate histogram
    hist, bins = np.histogram(differences, bins=bins)

    # Define x axis labels
    x_labels = ['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']

    # Plot histogram
    plt.bar(x_labels, hist, color='skyblue')

    # Set labels and title
    plt.xlabel('Time Range of Reviews')
    plt.ylabel('Number of Users')
    plt.title('Histogram of Time Range of Reviews')

    # Show plot
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ΜΕΤΑΤΡΕΠΕΙ ΤΙΣ ΜΕΡΕΣ ΣΕ ΧΡΟΝΙΑ, ΜΗΝΕΣ, ΜΕΡΕΣ
def format_time_difference(days):
    years = days // 365
    remaining_days = days % 365
    months = remaining_days // 30
    remaining_days %= 30
    return f"{years} years, {months} months, {remaining_days} days"

# ΥΠΟΛΟΓΙΖΕΙ ΤΟ ΧΡΟΝΙΚΟ ΕΥΡΟΣ ΑΞΙΟΛΟΓΗΣΕΩΝ (MAX-MIN) ΓΙΑ ΤΟΥΣ ΧΡΗΣΤΕΣ ΠΟΥ ΕΧΟΥΝ ΑΠΟ Rmin ΕΩΣ Rmax ΑΞΙΟΛΟΓΗΣΕΙΣ
# ΔΗΛΑΔΗ ΓΙΑ ΚΑΘΕ ΧΡΗΣΤΗ ΒΡΙΣΚΕΙ ΤΗ ΔΙΑΦΟΡΑ ΤΗΣ 1ης ΜΕ ΤΗΝ ΤΕΛΕΥΤΑΙΑ ΤΟΥ ΑΞΙΟΛΟΓΗΣΗ.
# ΤΟ ΠΕΡΙΟΡΙΣΜΕΝΟ ΣΥΝΟΛΟ ΧΡΗΣΤΩΝ ΠΟΥ ΕΧΟΥΝ ΑΠΟ Rmin ΕΩΣ Rmax ΑΞΙΟΛΟΓΗΣΕΙΣ ΒΡΙΣΚΕΤΑΙ ΣΤΟ 'filtered_df'
def time_ranges(filtered_df):
    time_differences = []

    for user in filtered_df['username'].unique():
        user_data = filtered_df[filtered_df['username'] == user]
        first_date = min(user_data['date'])
        last_date = max(user_data['date'])
        time_difference = (last_date - first_date).days
        time_differences.append((user, time_difference))

    print("\n\n*************************************")
    print("\nΧΡΟΝΙΚΟ ΕΥΡΟΣ ΑΞΙΟΛΟΓΗΣΕΩΝ ΤΟΥ ΠΕΡΙΟΡΙΣΜΕΝΟΥ ΣΥΝΟΛΟΥ ΧΡΗΣΤΩΝ:")
    print("\nTime Range for each user: \n", time_differences, "\n")
    print("Time ranges formated in years-months-days\n")

    for item in time_differences:
         print(item[0], format_time_difference(item[1]))

    return time_differences


def create_user_movie_array(filtered_df):
    # Αρχικοποίηση της κενής λίστας που θα περιέχει τα διανύσματα των βαθμολογιών για κάθε χρήστη
    user_movie_array = []
    # Δημιουργία λίστας με όλες τις μοναδικές ταινίες από το DataFrame
    unique_movies = filtered_df['movie'].unique()

    # Επανάληψη για κάθε μοναδικό χρήστη στο DataFrame
    for user in filtered_df['username'].unique():
        # Αρχικοποίηση της λίστας που θα περιέχει τις βαθμολογίες του τρέχοντος χρήστη για όλες τις ταινίες
        user_ratings = []

        # Φιλτράρισμα του DataFrame για να περιέχει μόνο τις γραμμές που αντιστοιχούν στον τρέχοντα χρήστη
        user_df = filtered_df[filtered_df['username'] == user]
        # Επανάληψη για κάθε μοναδική ταινία
        for movie in unique_movies:
            # Ανάκτηση της βαθμολογίας του χρήστη για την τρέχουσα ταινία, αν υπάρχει
            rating = user_df[user_df['movie'] == movie]['rating'].values
            # Έλεγχος αν υπάρχει βαθμολογία για την ταινία
            if len(rating) > 0:
                user_ratings.append(rating[0])
            else:
                user_ratings.append(0)
        user_movie_array.append(user_ratings)

    return user_movie_array

def create_binary_user_movie_array(R):
    # Δημιουργία νέου πίνακα με 1 όπου υπάρχει βαθμολογία και 0 όπου δεν υπάρχει
    binary_user_movie_array = [[1 if rating != 0 else 0 for rating in user_ratings] for user_ratings in R]
    return binary_user_movie_array

# ΟΜΑΔΟΠΟΙΗΣΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ(CLUSTERING) ΤΟΥ ΠΙΝΑΚΑ'R' ΜΕ ΤΟΝ ΑΛΓΟΡΙΘΜΟ 'K-Means' ΜΕ ΜΕΤΡΙΚΗ ΤΗΝ ΕΥΚΛΕΙΔΙΑ ΑΠΟΣΤΑΣΗ
def kmeans_clustering(R, L, metric='euclidean'):
    kmeans = KMeans(n_clusters=L, random_state=0)
    kmeans.fit(R)
    return kmeans.labels_


def k_means_for_clustering(R, L):
    # Εφαρμογή του αλγορίθμου k-means με διάφορες μετρικές απόστασης
    labels_euclidean = kmeans_clustering(R, L, metric='euclidean')
    labels_cosine = kmeans_clustering(R, L, metric='cosine')

    # Αναπαράσταση των συστάδων χρησιμοποιώντας PCA για μείωση της διαστατικότητας
    pca = PCA(n_components=2)

    # Μείωση της διαστατικότητας του πίνακα R
    R_reduced = pca.fit_transform(R)

    # Κλιμάκωση των τιμών στο εύρος [1, 10]
    min_value = np.min(R_reduced, axis=0)
    max_value = np.max(R_reduced, axis=0)
    scaled_data = 1 + (R_reduced - min_value) * 9 / (max_value - min_value)

    # Αναπαράσταση των συστάδων με PCA
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels_euclidean, cmap='viridis')
    plt.title('Clustering with Euclidean Distance')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.subplot(1, 2, 2)
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels_cosine, cmap='viridis')
    plt.title('Clustering with Cosine Similarity')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()


# ΕΠΙΣΤΕΡΦΕΙ ΓΙΑ ΚΑΘΕ ΧΡΗΣΤΗ ΑΠΟ ΤΟ ΠΕΡΙΟΡΙΣΝΕΝΟ ΣΥΝΟΛΟ(U ^) ΤΗ ΣΥΣΤΑΔΑ(CLUSTER) ΣΤΗΝ ΟΠΟΙΑ ΑΝΗΚΕΙ
def clustering_with_euclidean_distance(R, L,filtered_df):
    # Εφαρμογή του αλγορίθμου k-means
    cluster_labels = kmeans_clustering(R, L)

    # Λίστα με τα ονόματα των χρηστών
    usernames = filtered_df['username'].unique()

    # Δημιουργία λεξικού με τα ονόματα των χρηστών και τα αντίστοιχα cluster ids
    user_cluster_dict = {}
    for i, username in enumerate(usernames):
        user_cluster_dict[username] = cluster_labels[i] + 1  # Προσαρμόζουμε το cluster id στο εύρος [1, L]

    return user_cluster_dict


# ΕΠΙΣΤΕΡΦΕΙ ΓΙΑ ΚΑΘΕ ΧΡΗΣΤΗ ΑΠΟ ΤΟ ΠΕΡΙΟΡΙΣΝΕΝΟ ΣΥΝΟΛΟ(U ^) ΤΗ ΣΥΣΤΑΔΑ(CLUSTER) ΣΤΗΝ ΟΠΟΙΑ ΑΝΗΚΕΙ
def clustering_with_cosine_similarity(R, L,filtered_df):
    # Εφαρμογή του αλγορίθμου k-means
    cluster_labels = kmeans_clustering(R, L, metric='cosine')

    # Λίστα με τα ονόματα των χρηστών
    usernames = filtered_df['username'].unique()

    # Δημιουργία λεξικού με τα ονόματα των χρηστών και τα αντίστοιχα cluster ids
    user_cluster_dict = {}
    for i, username in enumerate(usernames):
        user_cluster_dict[username] = cluster_labels[i] + 1  # Προσαρμόζουμε το cluster id στο εύρος [1, L]

    return user_cluster_dict


def compute_distance(user1, user2):
    common_items = set(user1) & set(user2)
    union_items = set(user1) | set(user2)

    if len(common_items) == 0:
        return 1
    elif len(common_items) == len(union_items):
        return 0
    else:
        return 1 - len(common_items) / len(union_items)

def create_distance_matrix(R):
    n_users = len(R)
    D = np.zeros((n_users, n_users))
    for i in range(n_users):
        for j in range(n_users):
            if i != j:
                D[i][j] = compute_distance(R[i], R[j])
    return D

def spectral_clustering(R, L):
    D = create_distance_matrix(R)
    clustering = SpectralClustering(n_clusters=L, affinity='precomputed', random_state=0).fit(D)
    return clustering.labels_, D

def find_k_nearest_neighbors(D, user_index, k):
    # Assuming R is a numpy array representing the user-item matrix
    distances = np.linalg.norm(D - D[user_index], axis=1)  # Calculate distances
    nearest_indices = np.argsort(distances)[1:k+1]  # Exclude the user itself by starting from index 1
    nearest_distances = distances[nearest_indices]
    return nearest_indices, nearest_distances


def create_user_preference_vectors(D, user_indices, k):
    user_preference_vectors = []
    for user_index in user_indices:
        nearest_indices, _ = find_k_nearest_neighbors(D, user_index, k)
        user_preference_vector = []
        for neighbor_index in nearest_indices:
            user_preference_vector.append(D[neighbor_index])
        user_preference_vectors.append(user_preference_vector)
    return np.vstack(user_preference_vectors)  # Reshape to 2D array

def create_neural_network(X, y):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
    model.fit(X, y)
    return model

def train_neural_networks(D, user_indices, k):
    neural_networks = []
    for user_index in user_indices:
        X_all_users = create_user_preference_vectors(D, user_indices, k)  # Preference vectors for all users
        y = D[user_index]  # Target value for the current user
        X = X_all_users[user_index]  # Extract the preference vector for the current user
        neural_network = create_neural_network(X, y)
        neural_networks.append(neural_network)
    return neural_networks

def run():
    try:
        ################################################# Data From Dataset.npy #################################################
        # Load the dataset from dataset.npy
        dataset = np.load('dataset.npy')

        # Split each row of data into username, movie, rating, and date
        data_split = [row.split(',') for row in dataset]

        # Convert data into a pandas DataFrame
        df = pd.DataFrame(data_split, columns=['username', 'movie', 'rating', 'date'])

        #Convert types: username to string, movie to string, rating to int, date to datetime
        df['username'] = df['username'].astype(str)
        df['movie'] = df['movie'].astype(str)
        df['rating'] = df['rating'].astype(int)
        df['date'] = pd.to_datetime(df['date'])

################################################# Task 1 #################################################

        # ΒΡΕΙΤΕ ΤΟ ΣΥΝΟΛΟ ΤΩΝ ΜΟΝΑΔΙΚΩΝ ΧΡΗΣΤΩΝ U ΚΑΙ ΤΟ ΣΥΝΟΛΟ ΤΩΝ ΜΟΝΑΔΙΚΩΝ ΑΝΤΙΚΕΙΜΕΝΩΝ Ι
        find_unique_users_and_items(df)

################################################# Task 2 #################################################


        ratings_per_user = df.groupby('username').size()

        print("Rating per user:\n",ratings_per_user)
        print("***************************************\n")

        Rmin = 202  # Minimum number of ratings per user
        Rmax = 204  # Maximum number of ratings per user

        # ΒΡΙΣΚΩ ΤΟ ΣΥΝΟΛΟ ΧΡΗΣΤΩΝ ΠΟΥ ΕΧΟΥΝ ΑΠΟ Rmin ΕΩΣ Rmax ΑΞΙΟΛΟΓΉΣΕΙΣ
        filtered_users = ratings_per_user[(ratings_per_user >= Rmin) & (ratings_per_user <= Rmax)]
        print("\nFiltered users(",Rmin," to ",Rmax,"  reviews):\n",filtered_users)
        print("********************************************************")
        print("Number of filtered users: \n",len(filtered_users))

        filtered_df = question2(df,Rmin, Rmax, filtered_users)
        print("Number of filetered_df : ", len(filtered_df))

#################################################### Task 3 ################################################

        user_movies = [(row[0], row[1]) for row in data_split]

        # Count the number of unique movies each user has reviewed
        user_review_counts = {}
        for user, movie in user_movies:
            if user not in user_review_counts:
                user_review_counts[user] = 1
            else:
                user_review_counts[user] += 1
        #print("\nuser_review_counts: ",user_review_counts,"\n")

        # ΑΡΙΘΜΟΣ ΧΡΗΣΤΩΝ ΠΟΥ ΕΧΟΥΝ ΑΠΟ Rmin ΕΩΣ Rmax ΑΞΙΟΛΟΓΗΣΕΙΣ
        count_1_review = sum(1 for count in user_review_counts.values() if count == Rmin)
        count_2_reviews = sum(1 for count in user_review_counts.values() if count == int(Rmin+1))
        count_3_reviews = sum(1 for count in user_review_counts.values() if count == Rmax)
        print("\n\ncount_1_review: ",count_1_review)

        print("\nNumber of users with ",Rmin," reviews:", count_1_review)
        print("Number of users with ", int(Rmin+1), " reviews:", count_2_reviews)
        print("Number of users with ",Rmax," reviews:", count_3_reviews)
        print("\n***************************************")


        #  ΙΣΤΟΓΡΑΜΜΑ ΣΥΧΝΟΤΗΤΩΝ ΓΙΑ ΤΟ ΠΛΗΘΟΣ ΧΡΗΣΤΩΝ
        crowd_frequency_histogram(count_1_review,count_2_reviews,count_3_reviews)


        #  ΙΣΤΟΓΡΑΜΜΑ ΣΥΧΝΟΤΗΤΩΝ ΓΙΑ ΤΟ ΧΡΟΝΙΚΟ ΕΥΡΟΣ ΤΩΝ ΑΞΙΟΛΟΓΗΣΕΩΝ ΤΟΥ ΚΑΘΕ ΧΡΗΣΤΗ
        time_differences = time_ranges(filtered_df)
        plot_time_range_histogram(time_differences)

#################################################### Task 4 #################################################

    # ΓΙΑ ΤΟ ΠΕΡΙΟΡΙΣΜΕΝΟ ΣΥΝΟΛΟ ΧΡΗΣΤΩΝ ΠΟΥ ΕΧΕΙ ΑΞΙΟΛΟΓΗΣΕΙ ΣΥΓΚΕΚΡΙΜΕΝΟ ΣΥΝΟΛΟ ΤΑΙΝΙΩΝ(ΒΡΙΣΚΟΝΤΑΙ ΟΛΑ ΣΤΟ 'filtered_df')
    # ΔΗΜΙΟΥΡΓΕΙ ΤΟΝ ΠΙΝΑΚΑ R. ΟΠΟΙΑ ΤΑΙΝΙΑ ΔΕΝ ΕΧΕΙ ΒΑΘΜΟΛΟΓΗΘΕΙ ΑΠΟ ΤΟΝ ΧΡΗΣΤΗ ΒΑΖΟΥΜΕ '0'.

        users_from_df = filtered_df['username'].nunique()
        movies_from_df = filtered_df['movie'].nunique()
        print("Number of unique usernames in filtered_df:", users_from_df)
        print("\nNumber of unique movies in filtered_df:", movies_from_df)

        # ΔΗΜΙΟΥΡΓΙΑ ΠΙΝΑΚΑ R
        R = create_user_movie_array(filtered_df)
        num_elements_in_first_element = len(R[0])
        print("\nNumber of elements in first row:  ", num_elements_in_first_element)

        print("\nR=\n")
        print(R)

        # Δημιουργία του δυαδικού πίνακα
        binary_R = create_binary_user_movie_array(R)
        print("\nΔυαδικός πίνακας binary_R:")
        for row in binary_R:
            print(row)

######################################## Αλγόριθμοι Ομαδοποίησης Δεδομένων################################
#---------------------------------------     1ο Ερώτημα      ---------------------------------------------

        # ΟΡΙΖΩ ΤΟΝ ΑΡΙΘΜΟ ΤΩΝ ΣΥΣΤΑΔΩΝ(CLUSTERS)
        L = 3

        # ΕΠΙΣΤΡΕΦΕΙ ΓΙΑ ΚΑΘΕ ΧΡΗΣΤΗ ΑΠΟ ΤΟ ΠΕΡΙΟΡΙΣΜΕΝΟ ΣΥΝΟΛΟ U(^) ΤΗ ΣΥΣΤΑΔΑ(CLUSTER) ΠΟΥ ΑΝΗΚΕΙ
        user_cluster_euclidean = clustering_with_euclidean_distance(R, L,filtered_df)
        print("\nCluster for each user with Euclidean Distance: ")
        for record in user_cluster_euclidean.items():
            print(record)
        print("\n*******************************************************************")
        user_cluster_cosine = clustering_with_cosine_similarity(R, L,filtered_df)
        print("\nCluster for each user with Cosine Similarity: ")
        for record in user_cluster_cosine.items():
            print(record)

        # ΓΡΑΦΙΚΗ ΑΝΑΠΑΡΑΣΤΑΣΗ ΤΩΝ ΧΡΗΣΤΩΝ ΣΕ ΣΥΣΤΑΔΕΣ ΠΟΥ ΑΝΑΓΝΩΡΙΣΤΗΚΑΝ ΑΠΟ ΤΟΝ k-means ΑΛΓΟΡΙΘΜΟ
        k_means_for_clustering(R,L)

# ---------------------------------------     2ο Ερώτημα      ---------------------------------------------
        L2 = 3
        k = 5
        #Κλιμάκωση των χρηστών σε συστάδες
        clusters, D = spectral_clustering(R, L2)
        print("\nCluster for each user: ")
        i = 1
        for cluster in clusters:
            print("User ",i,"==",cluster)
            i+=1
        print("\nD: ", D)

        # Εκπαίδευση των νευρωνικών δικτύων για κάθε συστάδα
        neural_networks = train_neural_networks(D, range(len(D)), k)
        print("\n",neural_networks)

    except KeyboardInterrupt:
        print("\n\n*** Program terminated by user ***")





