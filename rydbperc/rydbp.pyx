#from libc.math cimport sin, cos, sqrt
cimport cython
cdef int all_connections(cluster, radius_shell, delta_shell):
    """
    this function returns a list of lists with all the possible connection between points.
    arg:
        cluster: np.array() 2xN_points.
        radius_shell: double. Radius of the facilitation shell.
        delta_shell: double. width of the facilitation shell. 
    
    returns:
        possible_connections: list of lists. the first index run on the points. the second index on all the connection with that point.
    """
    cdef int N = 0
    cdef double x_i, x_j, y_i, y_j, distance
    cdef list possible_connections = []
    for i in range(len(cluster[0,:])):
        possible_connections.append([])
        for j in range(i+1, len(cluster[0,:])):
            if (cluster[0,i]-cluster[0,j])>radius_shell:
                break
            x_i = cluster[0,i]*cos(cluster[1,i])
            x_j = cluster[0,j]*cos(cluster[1,j])
            y_i = cluster[0,i]*sin(cluster[1,i])
            y_j = cluster[0,j]*sin(cluster[1,j])
            distance = sqrt((x_i-x_j)**2 + (y_i - y_j)**2)
            if distance < radius_shell and distance > (radius_shell-delta_shell):
                possible_connections[i].append(j)
                N = N+1
    return possible_connections