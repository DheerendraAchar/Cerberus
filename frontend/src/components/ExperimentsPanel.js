import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart3, Trash2, Zap, Shield, ChevronLeft, ChevronRight, Search, Download, Star } from 'lucide-react';

export default function ExperimentsPanel() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [favorites, setFavorites] = useState([]);
  const itemsPerPage = 5;

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await axios.get('/api/experiments');
      setExperiments(response.data.experiments || []);
    } catch (error) {
      console.error('Error fetching experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleFavorite = (id) => {
    setFavorites(prev => 
      prev.includes(id) ? prev.filter(fav => fav !== id) : [...prev, id]
    );
  };

  const exportAsJSON = () => {
    const dataStr = JSON.stringify(experiments, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `experiments_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  const exportAsCSV = () => {
    const headers = ['ID', 'Attack Type', 'Architecture', 'Epsilon', 'Clean Accuracy', 'Adversarial Accuracy', 'Success Rate', 'Timestamp'];
    const rows = experiments.map(exp => [
      exp.id,
      exp.attack_type,
      exp.architecture,
      exp.epsilon.toFixed(4),
      exp.clean_accuracy?.toFixed(2),
      exp.adversarial_accuracy?.toFixed(2),
      exp.attack_success_rate?.toFixed(2),
      new Date(exp.timestamp).toLocaleString()
    ]);
    
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `experiments_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
  };

  const filteredExperiments = experiments.filter(exp => {
    const matchesSearch = exp.attack_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.architecture.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || exp.attack_type.toLowerCase().includes(filterType);
    return matchesSearch && matchesFilter;
  });

  const deleteExperiment = async (id) => {
    if (!window.confirm('Delete this experiment?')) return;
    try {
      await axios.delete(`/api/experiments/${id}`);
      fetchExperiments();
    } catch (error) {
      console.error('Error deleting experiment:', error);
    }
  };

  if (loading) {
    return <div className="panel">Loading experiments...</div>;
  }

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <BarChart3 size={24} style={{ color: '#667eea' }} />
        <h2 style={{ margin: 0 }}>Experiment History</h2>
      </div>

      {/* Search and Filter */}
      {experiments.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '12px', marginBottom: '20px' }}>
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: '#667eea' }} />
            <input
              type="text"
              placeholder="Search by attack or architecture..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              style={{
                width: '100%',
                padding: '10px 12px 10px 36px',
                backgroundColor: '#181c24',
                border: '1px solid #1e2430',
                borderRadius: '6px',
                color: '#e2e8f0',
                fontSize: '14px',
                outline: 'none'
              }}
            />
          </div>
          
          <select
            value={filterType}
            onChange={(e) => {
              setFilterType(e.target.value);
              setCurrentPage(1);
            }}
            style={{
              padding: '10px 12px',
              backgroundColor: '#181c24',
              border: '1px solid #1e2430',
              borderRadius: '6px',
              color: '#e2e8f0',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              minWidth: '120px'
            }}
          >
            <option value="all">All Types</option>
            <option value="fgsm">FGSM</option>
            <option value="pgd">PGD</option>
            <option value="cw">C&W</option>
            <option value="deepfool">DeepFool</option>
            <option value="jsma">JSMA</option>
          </select>

          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={exportAsJSON}
              style={{
                padding: '10px 12px',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                border: '1px solid rgba(102, 126, 234, 0.3)',
                borderRadius: '6px',
                color: '#667eea',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '13px',
                fontWeight: '600',
                transition: 'all 0.2s'
              }}
              title="Export as JSON"
            >
              <Download size={14} />
              JSON
            </button>
            <button
              onClick={exportAsCSV}
              style={{
                padding: '10px 12px',
                backgroundColor: 'rgba(168, 255, 120, 0.1)',
                border: '1px solid rgba(168, 255, 120, 0.3)',
                borderRadius: '6px',
                color: '#a8ff78',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '13px',
                fontWeight: '600',
                transition: 'all 0.2s'
              }}
              title="Export as CSV"
            >
              <Download size={14} />
              CSV
            </button>
          </div>
        </div>
      )}

      {experiments.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '60px 20px', color: '#999' }}>
          <BarChart3 size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
          <p>No experiments yet. Run an attack to get started!</p>
        </div>
      ) : (
        <>
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'grid', gap: '16px' }}>
              {filteredExperiments
                .slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)
                .map((exp, idx) => (
                  <div key={exp.id} style={{
                    backgroundColor: '#f8f9fa',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    padding: '16px',
                    transition: 'all 0.3s ease'
                  }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr auto', gap: '16px', alignItems: 'center' }}>
                      {/* Icon */}
                      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <div style={{
                          width: '40px',
                          height: '40px',
                          borderRadius: '8px',
                          backgroundColor: exp.attack_type.toLowerCase().includes('defense') ? 'rgba(46, 204, 113, 0.2)' : 'rgba(102, 126, 234, 0.2)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          {exp.attack_type.toLowerCase().includes('defense') ? 
                            <Shield size={20} style={{ color: '#2ecc71' }} /> : 
                            <Zap size={20} style={{ color: '#667eea' }} />
                          }
                        </div>
                      </div>

                  {/* Details */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'auto auto auto 1fr', gap: '24px', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Attack</div>
                      <div style={{ fontWeight: '600', color: '#333' }}>{exp.attack_type.toUpperCase()}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Architecture</div>
                      <div style={{ fontWeight: '600', color: '#333' }}>{exp.architecture}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>ε</div>
                      <div style={{ fontWeight: '600', color: '#333' }}>{exp.epsilon.toFixed(4)}</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Timestamp</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>{new Date(exp.timestamp).toLocaleString()}</div>
                    </div>
                  </div>

                  {/* Accuracy Metrics */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'auto auto auto auto', gap: '12px', textAlign: 'right' }}>
                    <div style={{ backgroundColor: 'rgba(46, 204, 113, 0.1)', padding: '8px 12px', borderRadius: '6px' }}>
                      <div style={{ fontSize: '11px', color: '#666', marginBottom: '2px' }}>Clean</div>
                      <div style={{ fontWeight: '700', color: '#2ecc71', fontSize: '14px' }}>
                        {exp.clean_accuracy?.toFixed(1)}%
                      </div>
                    </div>
                    <div style={{ backgroundColor: 'rgba(231, 76, 60, 0.1)', padding: '8px 12px', borderRadius: '6px' }}>
                      <div style={{ fontSize: '11px', color: '#666', marginBottom: '2px' }}>Adversarial</div>
                      <div style={{ fontWeight: '700', color: '#e74c3c', fontSize: '14px' }}>
                        {exp.adversarial_accuracy?.toFixed(1)}%
                      </div>
                    </div>
                    <div style={{ backgroundColor: 'rgba(155, 89, 182, 0.1)', padding: '8px 12px', borderRadius: '6px' }}>
                      <div style={{ fontSize: '11px', color: '#666', marginBottom: '2px' }}>Success</div>
                      <div style={{ fontWeight: '700', color: '#9b59b6', fontSize: '14px' }}>
                        {exp.attack_success_rate?.toFixed(1)}%
                      </div>
                      <button
                        onClick={() => toggleFavorite(exp.id)}
                        style={{
                          padding: '8px 12px',
                          backgroundColor: favorites.includes(exp.id) ? 'rgba(255, 209, 102, 0.2)' : 'rgba(155, 89, 182, 0.1)',
                          border: `1px solid ${favorites.includes(exp.id) ? 'rgba(255, 209, 102, 0.3)' : 'rgba(155, 89, 182, 0.2)'}`,
                          borderRadius: '6px',
                          cursor: 'pointer',
                          color: favorites.includes(exp.id) ? '#ffd166' : '#9b59b6',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '12px',
                          fontWeight: '600',
                          transition: 'all 0.2s'
                        }}
                        title={favorites.includes(exp.id) ? 'Remove from favorites' : 'Add to favorites'}
                      >
                        <Star size={14} fill={favorites.includes(exp.id) ? 'currentColor' : 'none'} />
                      </button>
                    </div>
                    <button
                      onClick={() => deleteExperiment(exp.id)}
                      style={{
                        padding: '8px 12px',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        border: '1px solid rgba(231, 76, 60, 0.2)',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        color: '#e74c3c',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        fontSize: '12px',
                        fontWeight: '600',
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.backgroundColor = 'rgba(231, 76, 60, 0.2)';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.backgroundColor = 'rgba(231, 76, 60, 0.1)';
                      }}
                    >
                      <Trash2 size={14} />
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
            </div>
          </div>

          {/* Pagination */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '24px', paddingTop: '16px', borderTop: '1px solid #e0e0e0' }}>
            <div style={{ color: '#666', fontSize: '14px' }}>
              Page {currentPage} of {Math.ceil(filteredExperiments.length / itemsPerPage)} ({filteredExperiments.length} results)
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                style={{
                  padding: '8px 12px',
                  backgroundColor: currentPage === 1 ? '#e0e0e0' : '#667eea',
                  color: currentPage === 1 ? '#999' : 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.2s'
                }}
              >
                <ChevronLeft size={16} />
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(Math.ceil(filteredExperiments.length / itemsPerPage), currentPage + 1))}
                disabled={currentPage === Math.ceil(filteredExperiments.length / itemsPerPage)}
                style={{
                  padding: '8px 12px',
                  backgroundColor: currentPage === Math.ceil(filteredExperiments.length / itemsPerPage) ? '#e0e0e0' : '#667eea',
                  color: currentPage === Math.ceil(filteredExperiments.length / itemsPerPage) ? '#999' : 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: currentPage === Math.ceil(experiments.length / itemsPerPage) ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.2s'
                }}
              >
                Next
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </>
      )}

      <div style={{ marginTop: '20px', padding: '15px', backgroundColor: 'rgba(0,229,255,0.05)', borderRadius: '6px', color: 'var(--muted)', border: '1px solid rgba(0,229,255,0.15)' }}>
        <strong>Total Experiments:</strong> {experiments.length}
        {experiments.length > 0 && (
          <>
            <br />
            <strong>Latest:</strong> {new Date(experiments[0].timestamp).toLocaleString()}
          </>
        )}
      </div>
    </div>
  );
}
