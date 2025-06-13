import numpy as np

def xor_data():
    x = np.array([[0,0], [0,1], [1,0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    return x, targets

def generate_circle_data(nb_points=1000, radius=0.5):
    x = np.random.uniform(-1, 1, (nb_points, 2))
    targets = (np.sum(x**2, axis=1) < radius**2).astype(int).reshape(-1, 1)
    return x, targets

def generate_spiral_data(nb_points=1000, noise=0.1):
    n = nb_points // 2
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r = theta

    # Spirale 1 (classe 0)
    x1 = np.array([r * np.cos(theta), r * np.sin(theta)]).T
    x1 += np.random.randn(n, 2) * noise
    y1 = np.zeros((n, 1))

    # Spirale 2 (classe 1) : opposée
    x2 = np.array([-r * np.cos(theta), -r * np.sin(theta)]).T
    x2 += np.random.randn(n, 2) * noise
    y2 = np.ones((n, 1))

    x = np.vstack((x1, x2))
    targets = np.vstack((y1, y2))
    return x, targets

def generate_cross_data(nb_img=1000, image_size=4):
    x_data = []
    y_data = []

    for _ in range(nb_img):
        img = np.zeros((image_size, image_size))
        label = 0

        if np.random.rand() > 0.5:
            # Genere croix
            cx = np.random.randint(1, image_size - 1)
            cy = np.random.randint(1, image_size - 1)
            img[cx, :] = 1
            img[:, cy] = 1
            label = 1
        else:
            # img aleatoire
            img = np.random.randint(0, 2, (image_size, image_size))

            # verifie croix accidentelle
            if image_size > 2:
                for i in range(1, image_size - 1):
                    if np.all(img[i, :] == 1) and np.all(img[:, i] == 1):
                        img[i, :] = 0
                        img[:, i] = 0

        x_data.append(img.flatten())
        y_data.append([label])

    return np.array(x_data), np.array(y_data)

def generate_patterns_data(nb_img=1000, image_size=4, n_labels=3):
    """
        Generates different patterns with different labels:
        -0: random noise
        -1: square
        -2: diagonal cross
        -3: triangle

    """
    def generate_square(img, size=5):
        cx = np.random.randint(1, image_size - size - 1)
        cy = np.random.randint(1, image_size - size - 1)
        img[cx, cy:cy+size-1] = 1
        img[cx+size-1, cy:cy+size-1] = 1
        img[cx:cx+size-1, cy] = 1
        img[cx:cx+size, cy+size-1] = 1
        return img
    
    def generate_cross(img, size=5):
        cx = np.random.randint(1, image_size - size - 1)
        cy = np.random.randint(1, image_size - size - 1)
        for i in range(size):
            x, y = cx+i, cy+i
            x2, y2 = cx + size -i, cy +i
            img[x, y] = 1
            img[x2, y2] = 1
        return img
    
    def generate_triangle(img, size=5):
        cx = np.random.randint(1, image_size - size - 1)
        cy = np.random.randint(1, image_size - size - 1)
        orientation = np.random.choice(['up_left', 'down_left', 'up_right', 'down_right'])
        for i in range(size):
            if orientation == "up_left":
                img[cx +i, cy] = 1
                img[cx, cy+i] = 1
                img[cx+i, cy+size-i-1] = 1
            elif orientation == "down_left":
                if i==0:
                    cx += size
                img[cx -i, cy] = 1
                img[cx, cy+i] = 1
                img[cx-i, cy+size-i-1] = 1
            elif orientation == "up_right":
                if i==0:
                    cy += size
                img[cx +i, cy] = 1
                img[cx, cy-i] = 1
                img[cx-i+size-1, cy-i] = 1
            else:
                if i==0:
                    cy += size
                    cx += size
                img[cx -i, cy] = 1
                img[cx, cy-i] = 1
                img[cx-size+i+1, cy-i] = 1
        return img
    
    n_labels = max(1, min(n_labels, 4))
    x, targets = [], []
    
    for img in range(nb_img):
        img = np.zeros((image_size, image_size))
        label = np.random.randint(n_labels) 
        
        
        if label == 0:
            img = np.random.randint(0, 2, (image_size, image_size))
        elif label == 1:
            img = generate_square(img, np.random.randint(5, image_size-2))
        elif label == 2:
            img = generate_cross(img, np.random.randint(5, image_size-2))
        elif label == 3:
            img = generate_triangle(img, np.random.randint(5, image_size-2))
        x.append(img.flatten())
        targets.append([label])

    return np.array(x), np.array(targets)