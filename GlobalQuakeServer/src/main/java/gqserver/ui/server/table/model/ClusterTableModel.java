package gqserver.ui.server.table.model;

import globalquake.core.earthquake.data.Cluster;
import gqserver.ui.server.table.Column;
import gqserver.ui.server.table.TableCellRendererAdapter;

import java.util.List;

public class ClusterTableModel extends FilterableTableModel<Cluster>{
    private final List<Column<Cluster, ?>> columns = List.of(
            Column.readonly("ID", Integer.class, Cluster::getId, new TableCellRendererAdapter<>()),
            Column.readonly("Assigned Events", Integer.class, cluster -> cluster.getAssignedEvents().size(), new TableCellRendererAdapter<>()),
            Column.readonly("level", Integer.class, Cluster::getActualLevel, new TableCellRendererAdapter<>()),
            Column.readonly("rootLat", Double.class, Cluster::getRootLat, new TableCellRendererAdapter<>()),
            Column.readonly("rootLon", Double.class, Cluster::getRootLon, new TableCellRendererAdapter<>()));


    public ClusterTableModel(List<Cluster> data) {
        super(data);
    }

    @Override
    public int getColumnCount() {
        return columns.size();
    }

    @Override
    public String getColumnName(int columnIndex) {
        return columns.get(columnIndex).getName();
    }

    public TableCellRendererAdapter<?, ?> getColumnRenderer(int columnIndex) {
        return columns.get(columnIndex).getRenderer();
    }

    @Override
    public Class<?> getColumnClass(int columnIndex) {
        return columns.get(columnIndex).getColumnType();
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columns.get(columnIndex).isEditable();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        Cluster event = getEntity(rowIndex);
        return columns.get(columnIndex).getValue(event);
    }

    @Override
    public void setValueAt(Object value, int rowIndex, int columnIndex) {
        Cluster event = getEntity(rowIndex);
        columns.get(columnIndex).setValue(value, event);
    }
}
