package gqserver.ui.server.table.model;

import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.earthquake.quality.QualityClass;
import gqserver.ui.server.table.Column;
import gqserver.ui.server.table.LastUpdateRenderer;
import gqserver.ui.server.table.TableCellRendererAdapter;

import java.time.LocalDateTime;
import java.util.List;

public class EarthquakeTableModel extends FilterableTableModel<Earthquake>{
    private final List<Column<Earthquake, ?>> columns = List.of(
            Column.readonly("Origin", LocalDateTime.class, Earthquake::getOriginDate, new LastUpdateRenderer<>()),
            Column.readonly("Region", String.class, Earthquake::getRegion, new TableCellRendererAdapter<>()),
            Column.readonly("Magnitude", Double.class, Earthquake::getMag, new TableCellRendererAdapter<>()),
            Column.readonly("Depth", Double.class, Earthquake::getDepth, new TableCellRendererAdapter<>()),
            Column.readonly("Lat", Double.class, Earthquake::getLat, new TableCellRendererAdapter<>()),
            Column.readonly("Lon", Double.class, Earthquake::getLon, new TableCellRendererAdapter<>()),
            Column.readonly("Quality", QualityClass.class, earthquake -> earthquake.getHypocenter().quality.getSummary(), new TableCellRendererAdapter<>()),
            Column.readonly("Revision", Integer.class, Earthquake::getRevisionID, new TableCellRendererAdapter<>()));


    public EarthquakeTableModel(List<Earthquake> data) {
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
        Earthquake event = getEntity(rowIndex);
        return columns.get(columnIndex).getValue(event);
    }

    @Override
    public void setValueAt(Object value, int rowIndex, int columnIndex) {
        Earthquake event = getEntity(rowIndex);
        columns.get(columnIndex).setValue(value, event);
    }
}
