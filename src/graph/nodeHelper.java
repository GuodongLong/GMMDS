package graph;

import java.util.ArrayList;
import java.util.List;

import Share.Comm;
import Share.log;

public class nodeHelper {

	/**
	 * find all shortest path from one node to other nodes.
	 * @return
	 */
	public static double[] SetShortestDistance(Node src, Graph graph)
	{
		double[] distances = new double[graph.nodeSize()];
		
		List<Node> extNodes = new ArrayList<Node>();
		extNodes.addAll(graph.nodes);
		
		List<Node> newNodes = new ArrayList<Node>();
		newNodes.add(src);
		extNodes.remove(src);
		
		int deepth = 0;
		while(extNodes.size() > 0)
		{
			if (newNodes.size() == 0)
			{
				break;
			}
			List<Node> newNodesTmp = new ArrayList<Node>();
			for(Node node : newNodes)
			{
				distances[node.docId] = deepth; // the shortest path length equals to the deepth.
				for(Node ne : node.neighbors)
				{
					if(extNodes.contains(ne)) // if the neighbour is an external node, then add it into internal graph.
					{
						newNodesTmp.add(ne);
						extNodes.remove(ne); // delete nodes from external graph
					}
				}
			}
			newNodes.clear();
			newNodes = newNodesTmp; // save it for the next iteration
			deepth++;
		}	

		// if there are some external nodes left, then they are unreachable to current source node.
		for(Node node: extNodes)
		{
			distances[node.docId] = 2 * graph.nodeSize();//Comm.INFINITE_DISTANCE;
		}
		
		return distances;
	}
	
	/**
	 * Get the number of shared words between two documents / nodes.
	 * @param node1
	 * @param node2
	 * @return
	 */
	public static int sharedWords(Node node1, Node node2)
	{
		int sharedCnt = 0;
		int i = 0, j = 0;
		while(true)
		{
			if (i >= node1.occurrence.size() || j >= node2.occurrence.size())
			{
				break;
			}
			
			int attr1 = node1.occurrence.get(i);
			int attr2 = node2.occurrence.get(j);
			
			if (attr1 < attr2)
			{
				i++;
				continue;
			}
			else if (attr1 > attr2)
			{
				j++;
				continue;
			}
			else //if (attr1 == attr2)
			{
				i++;
				j++;
				sharedCnt++;
				continue;
			}
		}
		
		return sharedCnt;
	}
	
}
