camera_height = 2562 #centimeters
ball_width = 61 #centimeters
focal_length = 732 #pixels

def calculate_ball_height(pixel_diameter):
    if pixel_diameter <= 0:
        raise ValueError("El diámetro en píxeles debe ser mayor que 0.")


    new_height = focal_length * (ball_width / pixel_diameter)

    # La distancia al suelo es la altura de la cámara menos Z
    distance_to_floor = camera_height - new_height

    return distance_to_floor