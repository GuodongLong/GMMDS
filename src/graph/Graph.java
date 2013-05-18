package graph;

import java.util.ArrayList;
import java.util.List;

public class Graph {

	public List<Node> nodes = new ArrayList<Node>();
	public List<Edge> edges = new ArrayList<Edge>();
	public int numAttribute = 0;
	
	public int nodeSize()
	{
		return nodes.size();
	}
	
	public int edgeSize()
	{
		return edges.size();
	}
	
	public int attrSize()
	{
		return numAttribute;
	}
	
	public Node getNode(int index)
	{
		return nodes.get(index);
	}
}
