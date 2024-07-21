import torch

def safe_normalize(x, eps=1e-6):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


# reference to https://github.com/fraunhoferhhi/neural-deferred-shading
class Mesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        vertex_normal (tensor): Vertex normal buffer (Vx3)
        device (torch.device): Device where the mesh buffers are stored
    """
    def __init__(self, vertices, indices,vertex_normals, device='cpu'):
        self.device = device

        self.vertices = vertices.to(device, dtype=torch.float32) if torch.is_tensor(vertices) else torch.tensor(vertices, dtype=torch.float32, device=device)
        self.indices = indices.to(device, dtype=torch.int64) if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.int64, device=device) if indices is not None else None
        if self.indices is not None:
            self.compute_normals()
        if vertex_normals is not None:
            self.vertex_normals = torch.tensor(vertex_normals, dtype=torch.float32, device=device)


    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """

        assert len(vertices) == len(self.vertices)
        mesh_new = Mesh(vertices, self.indices,None, self.device)
        return mesh_new

    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices][:, 0, :]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 
        # self.face_normals = torch.nn.functional.normalize(torch.cross(c - a, b - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals)
        # self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 
        self.vertex_normals = safe_normalize(vertex_normals) 

