package graph;

public class Edge {

	public Node node1, node2;
	public int sharedWords = 0;;
	
	public Edge(Node n1, Node n2, int sharedCnt)
	{
		this.node1 = n1;
		this.node2 = n2;
		this.sharedWords = sharedCnt;
	}
}
