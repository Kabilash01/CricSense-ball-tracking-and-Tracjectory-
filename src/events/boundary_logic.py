def intersects(ball_box, boundary_box, margin=5):
    bx1, by1, bx2, by2 = ball_box
    lx1, ly1, lx2, ly2, _ = boundary_box

    # expand boundary slightly for tolerance
    lx1 -= margin
    ly1 -= margin
    lx2 += margin
    ly2 += margin

    # AABB intersection
    if bx2 < lx1 or bx1 > lx2:
        return False
    if by2 < ly1 or by1 > ly2:
        return False

    return True
